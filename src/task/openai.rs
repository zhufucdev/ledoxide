use std::borrow::Cow;
use std::io::{Cursor, Read as _};

use async_openai::Client;
use async_openai::config::OpenAIConfig;
use async_openai::types::chat::{
    ChatCompletionRequestMessageContentPartTextArgs, ChatCompletionRequestUserMessageArgs,
    ChatCompletionRequestUserMessageContent, ChatCompletionRequestUserMessageContentPart,
    CreateChatCompletionRequestArgs, ImageUrlArgs, ReasoningEffort, ResponseFormat,
    ResponseFormatJsonSchema,
};
use axum::RequestExt;
use axum_extra::headers::Mime;
use base64::Engine;
use base64::prelude::BASE64_STANDARD;
use schemars::{JsonSchema, json_schema, schema_for};
use serde::Deserialize;
use smol_str::{SmolStr, ToSmolStr};
use tracing::{Level, event};
use zip::ZipArchive;
use zip::result::ZipError;

use crate::bill::Category;
use crate::ext::{ExtractImageBuf as _, FromEnvVars};
use crate::{
    bill::Bill,
    error::{CreateTaskError, RunTaskError},
    task::{RunTask, TaskDescriptor},
};

pub const OPENAI_VISION_MODEL: &str = "gpt-4o";
pub const OPENAI_TEXT_MODEL: &str = "gpt-4o-mini";

#[derive(Debug, Clone)]
pub struct OpenAIRunTask {
    pub client: Client<OpenAIConfig>,
    pub caption_model: SmolStr,
    pub extract_model: SmolStr,
}

impl Default for OpenAIRunTask {
    fn default() -> Self {
        Self {
            client: Client::from_env_vars(),
            caption_model: OPENAI_VISION_MODEL.into(),
            extract_model: OPENAI_TEXT_MODEL.into(),
        }
    }
}

impl OpenAIRunTask {
    fn image_part(&self, buf: &[u8]) -> ChatCompletionRequestUserMessageContentPart {
        let data_url = format!("data:image/jpeg;base64,{}", BASE64_STANDARD.encode(buf));
        let image_part = ImageUrlArgs::default()
            .url(data_url)
            .build()
            .unwrap()
            .into();
        ChatCompletionRequestUserMessageContentPart::ImageUrl(image_part)
    }

    async fn chat_with_images(
        &self,
        model: &str,
        images: &[&[u8]],
        text: &str,
        response_format: Option<ResponseFormat>,
        reasoning_effort: ReasoningEffort,
    ) -> Result<String, RunTaskError> {
        let mut parts: Vec<ChatCompletionRequestUserMessageContentPart> =
            images.iter().map(|buf| self.image_part(buf)).collect();
        parts.push(ChatCompletionRequestUserMessageContentPart::Text(
            ChatCompletionRequestMessageContentPartTextArgs::default()
                .text(text.to_string())
                .build()
                .unwrap(),
        ));

        let user_message = ChatCompletionRequestUserMessageArgs::default()
            .content(ChatCompletionRequestUserMessageContent::Array(parts))
            .build()?
            .into();

        let mut request = CreateChatCompletionRequestArgs::default()
            .model(model)
            .messages([user_message])
            .reasoning_effort(reasoning_effort)
            .build()?;

        if let Some(format) = response_format {
            request.response_format = Some(format);
        }

        let response = self
            .client
            .chat()
            .create(request)
            .await
            .map_err(|err| RunTaskError::Runner(err.into()))?;

        response
            .choices
            .first()
            .and_then(|c| c.message.content.clone())
            .ok_or_else(|| RunTaskError::InvalidOutput("empty response".into()))
    }

    async fn chat_text(
        &self,
        model: &str,
        text: &str,
        response_format: Option<ResponseFormat>,
        reasoning_effort: ReasoningEffort,
    ) -> Result<String, RunTaskError> {
        self.chat_with_images(model, &[], text, response_format, reasoning_effort)
            .await
    }
}

impl RunTask for OpenAIRunTask {
    type TaskDescriptor = OpenAITaskDescriptor;

    async fn extract(&self, task: &Self::TaskDescriptor) -> Result<Bill, RunTaskError> {
        tracing::debug!("extract, caption_model: {:?}", self.caption_model);
        // Step 1: Generate caption
        let prompt = include_str!("../../prompt/description.md");
        let caption = self
            .chat_with_images(
                &self.caption_model,
                &task.images(),
                prompt,
                None,
                ReasoningEffort::None,
            )
            .await?;
        event!(Level::DEBUG, "caption: {}", caption);

        // Step 2: Generate structured notes
        let notes_prompt = format!(include_str!("../../prompt/note_taking.md"), caption);
        let notes_schema = schema_for!(Notes);
        let notes_response = self
            .chat_with_images(
                &self.caption_model,
                &task.images(),
                &notes_prompt,
                Some(ResponseFormat::JsonSchema {
                    json_schema: ResponseFormatJsonSchema {
                        description: Some("Purchase notes".into()),
                        name: "notes".into(),
                        schema: serde_json::to_value(&notes_schema)
                            .map_err(|err| RunTaskError::Runner(err.into()))?,
                        strict: Some(true),
                    },
                }),
                ReasoningEffort::Medium,
            )
            .await?;
        event!(Level::DEBUG, "notes: {}", notes_response);

        let notes =
            if let Ok(structured_notes) = serde_json::from_str::<Notes>(notes_response.as_str()) {
                structured_notes.to_string()
            } else {
                event!(Level::WARN, "invalid notes JSON: {}", notes_response);
                notes_response
            };

        // Step 3 & 4: Extract amount and category in parallel
        #[derive(JsonSchema, Deserialize)]
        struct Amount {
            amount: f32,
        }

        #[derive(JsonSchema, Deserialize)]
        struct Category {
            category: Option<String>,
        }

        let amount_schema = schema_for!(Amount);
        let category_schema = json_schema!({
            "description": "Category of the goods",
            "type": "object",
            "properties": {
                "category": {
                    "enum": task.category_names()
                }
            },
        });

        let amount_response = self
            .chat_text(
                &self.extract_model,
                &format!(
                    include_str!("../../prompt/amount_extraction.md"),
                    notes, caption
                ),
                Some(ResponseFormat::JsonSchema {
                    json_schema: ResponseFormatJsonSchema {
                        description: Some("Final payment amount".into()),
                        name: "amount".into(),
                        schema: serde_json::to_value(&amount_schema)
                            .map_err(|err| RunTaskError::Runner(err.into()))?,
                        strict: Some(true),
                    },
                }),
                ReasoningEffort::Medium,
            )
            .await?;
        event!(Level::DEBUG, "amount: {}", amount_response);

        let category_response = self
            .chat_text(
                &self.extract_model,
                &format!(
                    include_str!("../../prompt/categorization.md"),
                    notes,
                    caption,
                    task.category_names()
                        .iter()
                        .map(|c| format!("- {}", c))
                        .collect::<Vec<_>>()
                        .join("\n")
                ),
                Some(ResponseFormat::JsonSchema {
                    json_schema: ResponseFormatJsonSchema {
                        description: Some("Best matching category".into()),
                        name: "category".into(),
                        schema: serde_json::to_value(&category_schema)
                            .map_err(|err| RunTaskError::Runner(err.into()))?,
                        strict: Some(true),
                    },
                }),
                ReasoningEffort::Medium,
            )
            .await?;
        event!(Level::DEBUG, "category: {}", category_response);

        let structured_amount = serde_json::from_str::<Amount>(&amount_response)
            .map_err(|_| RunTaskError::InvalidOutput("price".into()))?;
        let structured_category = serde_json::from_str::<Category>(&category_response)
            .map_err(|_| RunTaskError::InvalidOutput("category".into()))?;

        Ok(Bill {
            notes: notes.into(),
            amount: structured_amount.amount,
            category: structured_category.category.map(|n| n.into()),
        })
    }
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct OpenAITaskDescriptor {
    images_buf: Box<[Box<[u8]>]>,
    categories: Option<Box<[SmolStr]>>,
}

impl TaskDescriptor for OpenAITaskDescriptor {
    fn images(&self) -> Box<[&[u8]]> {
        self.images_buf
            .iter()
            .map(|buf| buf.as_ref())
            .collect::<Box<_>>()
    }

    fn category_names(&self) -> Box<[SmolStr]> {
        self.categories.clone().unwrap_or_else(|| {
            Category::all_cases()
                .iter()
                .map(|c| c.name())
                .collect::<Box<_>>()
        })
    }
}

impl<S> axum::extract::FromRequest<S> for OpenAITaskDescriptor
where
    S: Send + Sync,
{
    type Rejection = CreateTaskError;

    async fn from_request(req: axum::extract::Request, _: &S) -> Result<Self, Self::Rejection> {
        use axum::body::Bytes;
        use axum::extract::Multipart;

        let content_type = req
            .headers()
            .get("Content-Type")
            .and_then(|v| v.to_str().ok())
            .map(|v| v.to_string())
            .unwrap_or_default();
        event!(Level::DEBUG, "receiving {}", content_type);

        let mut images_buf = None;
        let mut categories = None;

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
                    "categories" => {
                        let value: Vec<String> =
                            serde_json::from_str(field.text().await?.as_str())?;
                        categories = Some(
                            value
                                .into_iter()
                                .map(|name| name.to_smolstr())
                                .collect::<Box<_>>(),
                        );
                    }
                    _ => {}
                }
            }
        } else {
            let buf: Bytes = req.extract().await?;
            let mime = content_type.parse()?;
            images_buf = Some((buf, mime).extract_image_buf()?);
        }

        if images_buf.is_none() {
            return Err(CreateTaskError::MissingField("image".to_string()));
        }

        Ok(Self {
            images_buf: images_buf.unwrap(),
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

impl std::fmt::Display for Notes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(retailer) = &self.retailer {
            write!(f, "{} \"{}\" from {}", self.type_, self.name, retailer)
        } else {
            write!(f, "{} \"{}\"", self.type_, self.name)
        }
    }
}
