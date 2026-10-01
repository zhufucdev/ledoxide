use axum::RequestExt;
use axum::extract::Multipart;
use axum_extra::headers::Mime;
use base64::Engine;
use base64::prelude::BASE64_STANDARD;
use encoding_rs::UTF_8;
use ollama_rs::Ollama;
use ollama_rs::error::OllamaError;
use ollama_rs::generation::completion::request::GenerationRequest;
use ollama_rs::generation::images::Image;
use ollama_rs::generation::parameters::{
    FormatType, JsonSchema, JsonStructure, KeepAlive, TimeUnit,
};
use ollama_rs::models::create::CreateModelRequest;
use schemars::json_schema;
use std::borrow::Cow;
use std::fmt::Display;
use std::io::{Cursor, Read};
use tracing::{Level, event};
use zip::result::ZipError;

use axum::{body::Bytes, extract::FromRequest};
use ollama_rs::models::ModelOptions;
use serde::Deserialize;
use smol_str::{SmolStr, ToSmolStr};
use zip::ZipArchive;

use crate::bill::{OwnedCategory, SharedCategory};
use crate::ext::{ExtractImageBuf, FromEnvVars};
use crate::task::ext::DisplayCategory;
use crate::{
    bill::Bill,
    error::{CreateTaskError, RunTaskError},
    task::{RunTask, TaskDescriptor},
};

#[derive(Debug, Clone)]
pub struct OllamaRunTask {
    pub ollama: Ollama,
    pub caption_model: SmolStr,
    pub extract_model: SmolStr,
    pub offline: bool,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct OllamaTaskDescriptor {
    images_buf: Box<[Box<[u8]>]>,
    lm_options: Option<ModelOptions>,
    vlm_options: Option<ModelOptions>,
    categories: Option<Box<[OwnedCategory]>>,
}

pub const GEMMA_4_E4B_Q4KM: &str = "gemma4:e4b";

impl Default for OllamaRunTask {
    fn default() -> Self {
        Self {
            ollama: Ollama::from_env_vars(),
            caption_model: GEMMA_4_E4B_Q4KM.into(),
            extract_model: GEMMA_4_E4B_Q4KM.into(),
            offline: false,
        }
    }
}

impl OllamaRunTask {
    pub async fn pull_models(&self) -> Result<(), OllamaError> {
        futures::future::try_join_all(
            [self.extract_model.clone(), self.caption_model.clone()]
                .into_iter()
                .map(async |model| -> Result<(), OllamaError> {
                    if !self
                        .ollama
                        .list_local_models()
                        .await?
                        .iter()
                        .any(|m| m.name == model)
                    {
                        if let Some((name, quant)) = model.split_once("/") {
                            let (name, quant) = if let Some((quant, size)) = quant.split_once(":") {
                                (format!("{name}:{size}"), quant)
                            } else {
                                (name.to_string(), quant)
                            };
                            self.ollama.pull_model(name.clone(), false).await?;
                            self.ollama
                                .create_model(
                                    CreateModelRequest::new(model.clone().into())
                                        .from_model(name)
                                        .quantize(serde_plain::from_str(quant).unwrap()),
                                )
                                .await?;
                        } else {
                            self.ollama.pull_model(model.into(), false).await?;
                        }
                    }
                    Ok(())
                }),
        )
        .await?;
        Ok(())
    }

    pub async fn unload_models(&self) -> Result<(), OllamaError> {
        self.ollama
            .generate(
                GenerationRequest::new(self.caption_model.clone().into(), "").keep_alive(
                    KeepAlive::Until {
                        time: 0,
                        unit: TimeUnit::Seconds,
                    },
                ),
            )
            .await?;
        self.ollama
            .generate(
                GenerationRequest::new(self.extract_model.clone().into(), "").keep_alive(
                    KeepAlive::Until {
                        time: 0,
                        unit: TimeUnit::Seconds,
                    },
                ),
            )
            .await?;
        Ok(())
    }
}

impl RunTask for OllamaRunTask {
    type TaskDescriptor = OllamaTaskDescriptor;

    async fn extract(&self, task: &Self::TaskDescriptor) -> Result<Bill, RunTaskError> {
        if !self.offline {
            event!(Level::INFO, "pulling models");
            self.pull_models().await?;
        }

        let prompt = include_str!("../../prompt/description.md");
        let ims = task
            .images()
            .iter()
            .map(|buf| BASE64_STANDARD.encode(buf))
            .map(Image::from_base64)
            .collect::<Vec<_>>();
        let caption = self
            .ollama
            .generate({
                let r = GenerationRequest::new(self.caption_model.clone().into(), prompt)
                    .images(ims.clone())
                    .think(true);
                if let Some(lm_options) = task.lm_options() {
                    r.options(lm_options.clone())
                } else {
                    r
                }
            })
            .await
            .map_err(|err| RunTaskError::Runner(err.into()))?;
        assert!(caption.done);
        event!(Level::DEBUG, "caption: {}", caption.response);
        let prompt = format!(
            include_str!("../../prompt/note_taking.md"),
            caption.response
        );
        let notes = self
            .ollama
            .generate({
                let r = GenerationRequest::new(self.caption_model.clone().into(), prompt)
                    .images(ims)
                    .think(true)
                    .format(FormatType::StructuredJson(Box::new(JsonStructure::new::<
                        Notes,
                    >(
                    ))));
                if let Some(options) = task.vlm_options() {
                    r.options(options.clone())
                } else {
                    r
                }
            })
            .await
            .map_err(|err| RunTaskError::Runner(err.into()))?;
        assert!(notes.done);
        event!(Level::DEBUG, "notes: {}", notes.response);
        #[derive(JsonSchema, Deserialize)]
        struct Amount {
            amount: f32,
        }
        #[derive(JsonSchema, Deserialize)]
        struct Category {
            category: Option<String>,
        }
        let notes = if let Ok(structured_notes) =
            serde_json::from_str::<Notes>(notes.response.as_str())
        {
            structured_notes.to_string()
        } else {
            event!(target: "ollama_run_task",Level::WARN,  "invalid notes JSON: {}", notes.response);
            notes.response
        };
        let category_schema = json_schema!({
            "description": "Category of the goods",
            "type": "object",
            "properties": {
                "category": {
                    "enum": task.category_names()
                }
            },
        });
        let (amount, category) = futures::try_join!(
            self.ollama.generate({
                let r = GenerationRequest::new(
                    self.extract_model.clone().into(),
                    format!(
                        include_str!("../../prompt/amount_extraction.md"),
                        notes, caption.response
                    ),
                )
                .think(true)
                .format(FormatType::StructuredJson(Box::new(JsonStructure::new::<
                    Amount,
                >())));
                if let Some(options) = task.lm_options() {
                    r.options(options.clone())
                } else {
                    r
                }
            },),
            self.ollama.generate({
                let r = GenerationRequest::new(
                    self.extract_model.clone().into(),
                    format!(
                        include_str!("../../prompt/categorization.md"),
                        notes,
                        caption.response,
                        task.categories()
                            .iter()
                            .map(|c| format!("{}", c.bullet_item()))
                            .collect::<Vec<_>>()
                            .join("\n")
                    ),
                )
                .think(true)
                .format(FormatType::StructuredJson(Box::new(
                    JsonStructure::new_for_schema(category_schema),
                )));
                if let Some(options) = task.lm_options() {
                    r.options(options.clone())
                } else {
                    r
                }
            })
        )
        .map_err(|err| RunTaskError::Runner(err.into()))?;
        event!(Level::DEBUG, "amount: {}", amount.response);
        event!(Level::DEBUG, "category: {}", category.response);
        let structured_amount = serde_json::from_str::<Amount>(amount.response.as_str())
            .map_err(|_| RunTaskError::InvalidOutput("price".into()))?;
        let structured_category = serde_json::from_str::<Category>(category.response.as_str())
            .map_err(|_| RunTaskError::InvalidOutput("category".into()))?;

        Ok(Bill {
            notes: notes.into(),
            amount: structured_amount.amount,
            category: structured_category.category.map(|n| n.into()),
        })
    }
}

impl TaskDescriptor for OllamaTaskDescriptor {
    type Category = OwnedCategory;

    fn images(&self) -> Box<[&[u8]]> {
        self.images_buf
            .iter()
            .map(|buf| buf.as_ref())
            .collect::<Box<_>>()
    }

    fn categories(&self) -> Box<[Self::Category]> {
        self.categories.clone().unwrap_or_else(|| {
            SharedCategory::all_cases()
                .iter()
                .map(|c| c.into())
                .collect::<Box<[_]>>()
        })
    }
}

impl OllamaTaskDescriptor {
    pub fn lm_options(&self) -> Option<&ModelOptions> {
        self.lm_options.as_ref()
    }

    pub fn vlm_options(&self) -> Option<&ModelOptions> {
        self.vlm_options.as_ref()
    }
}

impl<S> FromRequest<S> for OllamaTaskDescriptor
where
    S: Send + Sync,
{
    type Rejection = CreateTaskError;

    async fn from_request(req: axum::extract::Request, _: &S) -> Result<Self, Self::Rejection> {
        let content_type = UTF_8
            .decode(req.headers().get("Content-Type").unwrap().as_bytes())
            .0;
        event!(Level::DEBUG, "receiving {}", content_type);

        let mut images_buf = None;
        let (mut lm_options, mut vlm_options, mut categories) = (None, None, None);
        if content_type.starts_with("multipart/form-data") {
            let mut form: Multipart = req.extract().await?;
            while let Some(field) = form.next_field().await? {
                let name = field.name().unwrap().to_string();
                match name.as_str() {
                    "image" => {
                        let mime: Mime = field
                            .content_type()
                            .ok_or(CreateTaskError::UnspecificContentType("image".to_string()))?
                            .parse()?;
                        images_buf = Some((field.bytes().await?, mime).extract_image_buf()?);
                    }
                    "lm_options" | "vlm_options" => {
                        if let Some(mime) = field.content_type()
                            && mime != "application/json"
                        {
                            return Err(CreateTaskError::InvalidField(name.to_string()));
                        }
                        let value: ModelOptions =
                            serde_json::from_str(field.text().await?.as_str())?;
                        if name.starts_with("lm") {
                            lm_options = Some(value)
                        } else {
                            vlm_options = Some(value)
                        }
                    }
                    "categories" => {
                        if let Some(mime) = field.content_type()
                            && mime != "application/json"
                        {
                            return Err(CreateTaskError::InvalidField(name.to_string()));
                        }
                        let value: Vec<String> =
                            serde_json::from_str(field.text().await?.as_str())?;
                        categories = Some(
                            value
                                .into_iter()
                                .map(OwnedCategory::only_name)
                                .collect::<Box<_>>(),
                        );
                    }
                    _ => {
                        return Err(CreateTaskError::UnknownField(name.to_string()));
                    }
                }
            }
        } else {
            let mime: Mime = content_type.parse()?;
            let buf: Bytes = req.extract().await?;
            images_buf = Some((buf, mime).extract_image_buf()?);
        }
        if images_buf.is_none() {
            return Err(CreateTaskError::MissingField("image".to_string()));
        }

        Ok(Self {
            images_buf: images_buf.unwrap(),
            lm_options,
            vlm_options,
            categories,
        })
    }
}

#[derive(JsonSchema, Deserialize)]
struct Notes {
    name: String,
    #[schemars(rename = "type")]
    #[serde(rename = "type")]
    type_: String,
    retailer: Option<String>,
}

impl Display for Notes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(retailer) = &self.retailer {
            write!(f, "{} \"{}\" from {}", self.type_, self.name, retailer)
        } else {
            write!(f, "{} \"{}\"", self.type_, self.name)
        }
    }
}

#[cfg(test)]
mod tests {
    use tracing_test::traced_test;

    use super::*;

    #[tokio::test]
    #[traced_test]
    async fn test_extract_default_model() {
        let req = OllamaTaskDescriptor {
            images_buf: Box::new([]),
            lm_options: None,
            vlm_options: None,
            categories: Some(
                [
                    OwnedCategory::only_name("Shopping"),
                    OwnedCategory::only_name("Food"),
                    OwnedCategory::only_name("Transport"),
                    OwnedCategory::only_name("Rent"),
                ]
                .into(),
            ),
        };
        let runner = OllamaRunTask::default();
        let bill = runner.extract(&req).await.unwrap();
        event!(Level::INFO, "{:#?}", bill);
    }
}
