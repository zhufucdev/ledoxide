use std::{
    borrow::Cow,
    io::{Cursor, Read},
    sync::{Arc, RwLock},
};

use anyhow::anyhow;
use axum::{
    RequestExt,
    body::Bytes,
    extract::{FromRequest, Multipart},
};
use encoding_rs::UTF_8;
use image::EncodableLayout;
use regex::Regex;
use serde::{Deserialize, Serialize, ser::SerializeStruct};
use strum::Display;
use zip::{
    ZipArchive,
    result::ZipError,
    unstable::stream::{ZipStreamReader, ZipStreamVisitor},
};

use crate::{
    bill::{Bill, Category},
    error::{CreateTaskError, RunTaskError},
    key,
    models::TimedModel,
    runner::{
        ImageOrText, MessageRole, TextLmRequest, TextLmRunner, TextLmRunnerExt, VisionLmRequest,
        VisionLmRunner, VisionLmRunnerExt,
    },
    sample::{LlguidanceSamplingParams, LlguidanceSchema, SimpleSamplingParams},
};

#[derive(Debug, Clone, Deserialize, Default)]
pub struct TaskDescriptor {
    images_buf: Vec<Vec<u8>>,
    lm_sampling: Option<SimpleSamplingParams>,
    vlm_sampling: Option<SimpleSamplingParams>,
}

impl TaskDescriptor {
    pub fn images_count(&self) -> usize {
        self.images_buf.len()
    }

    pub fn image_bytes(&self, idx: usize) -> &[u8] {
        &self.images_buf[idx]
    }

    pub fn lm_sampling(&self) -> Option<SimpleSamplingParams> {
        self.lm_sampling.clone()
    }

    pub fn vlm_sampling(&self) -> Option<SimpleSamplingParams> {
        self.vlm_sampling.clone()
    }

    pub async fn run<VLM, LM>(
        &self,
        vlm: &TimedModel<VLM>,
        lm: Option<&TimedModel<LM>>,
    ) -> Result<Bill, RunTaskError>
    where
        for<'a> LM: TextLmRunner<'a> + Send + Sync + 'static,
        for<'a> VLM: VisionLmRunner<'a> + TextLmRunner<'a> + Send + Sync + 'static,
    {
        let description = self.get_description(vlm.get_model().await?.as_ref())?;
        log::debug!(target: "task runner", "description: {}", description);

        if let Some(lm) = lm {
            let model = lm.get_model().await?;
            self.run_with_description(model.as_ref(), &description)
        } else {
            let model = vlm.get_model().await?;
            self.run_with_description(model.as_ref(), &description)
        }
    }

    fn get_description(
        &self,
        runner: &impl for<'a> VisionLmRunner<'a>,
    ) -> Result<String, RunTaskError> {
        let mut request = VisionLmRequest {
            messages: self
                .images_buf
                .iter()
                .map(|buf| {
                    image::load_from_memory(buf.as_slice())
                        .map(|im| (MessageRole::User, ImageOrText::Image(im)))
                })
                .chain(std::iter::once(Ok((
                    MessageRole::User,
                    ImageOrText::Text(format!(include_str!("../prompt/description.md"))),
                ))))
                .collect::<Result<Vec<_>, _>>()?,
            prefill: Some("<think>\n".to_string()),
            ..Default::default()
        };
        if let Some(sampling) = &self.vlm_sampling {
            request.sampling = sampling.clone();
        }
        let mut output = runner.get_vlm_response(request)?;
        if output.contains("<think>") {
            // strip the <think> tag
            const END_TAG: &str = "</think>";
            if let Some(end) = output.find(END_TAG) {
                output = output.split_off(end + END_TAG.len())
            }
        }
        while output.chars().nth(0).map_or(false, |c| c.is_whitespace()) {
            output = output.trim_start().to_string();
        }
        Ok(output)
    }

    fn run_with_description<'a, LM>(
        &self,
        lm: &'a LM,
        description: &str,
    ) -> Result<Bill, RunTaskError>
    where
        LM: TextLmRunner<'a> + Send + Sync + 'static,
    {
        let mut request = TextLmRequest {
            messages: vec![(
                MessageRole::User,
                format!(include_str!("../prompt/note_taking.md"), description),
            )],
            llguidance: Some(LlguidanceSamplingParams {
                schema: LlguidanceSchema::Lark,
                data: include_str!("../constraint/note_taking.lark").to_string(),
            }),
            prefill: Some("Summary".to_string()),
            ..Default::default()
        };
        if let Some(sampling) = &self.lm_sampling {
            request.sampling = sampling.clone();
        }

        let notes = lm.get_lm_response(request)?;
        let notes = notes.trim();
        log::debug!(target: "lm", "{}", notes);
        let notes = notes
            .split_once(":")
            .ok_or(RunTaskError::InvalidOutput("notes".to_string()))?
            .1
            .trim_start()
            .to_string();
        log::debug!(target: "task runner", "notes: {}", notes);
        let mut request = TextLmRequest {
            messages: vec![(
                MessageRole::User,
                format!(
                    include_str!("../prompt/amount_extraction.md"),
                    notes, description
                ),
            )],
            llguidance: Some(LlguidanceSamplingParams {
                schema: LlguidanceSchema::Lark,
                data: include_str!("../constraint/amount_extraction.lark").to_string(),
            }),
            ..Default::default()
        };
        if let Some(sampling) = &self.lm_sampling {
            request.sampling = sampling.clone();
        }
        let response = lm.get_lm_response(request)?;
        log::debug!(target: "lm", "{}", response);
        let numeric = Regex::new(r#"([0-9,]+\.?[0-9]{0,})"#).unwrap();
        let amount: f32 = if let Some(amount) = numeric.captures(&response) {
            amount
                .get(1)
                .ok_or(RunTaskError::InvalidOutput("amount extraction".to_string()))?
                .as_str()
                .parse()
                .map_err(|_| RunTaskError::EmptyAmount(response.clone()))?
        } else {
            return Err(RunTaskError::EmptyAmount(response));
        };
        log::debug!(target: "task runner", "amount: {}", amount);

        let mut request = TextLmRequest {
            messages: vec![(
                MessageRole::User,
                format!(
                    include_str!("../prompt/categorization.md"),
                    notes,
                    description,
                    Category::all_cases()
                        .iter()
                        .map(|c| format!("- {}", c.name()))
                        .collect::<Vec<_>>()
                        .join("\n")
                ),
            )],
            llguidance: Some(LlguidanceSamplingParams {
                schema: LlguidanceSchema::Lark,
                data: format!(
                    include_str!("../constraint/categorization.lark"),
                    Category::all_cases()
                        .iter()
                        .map(|c| format!(r#""{}""#, c.name()))
                        .collect::<Vec<_>>()
                        .join("|")
                )
                .to_string(),
            }),
            prefill: Some("Category".to_string()),
            ..Default::default()
        };
        if let Some(sampling) = &self.lm_sampling {
            request.sampling = sampling.clone();
        }
        let response = lm.get_lm_response(request)?;
        log::debug!(target: "lm", "{}", response);
        let extract_category = Regex::new(r#"Category: (.+)\.?"#).unwrap();
        let category = extract_category
            .captures(&response)
            .ok_or(RunTaskError::InvalidOutput("categorization".to_string()))?
            .get(1)
            .unwrap()
            .as_str();

        Ok(Bill {
            notes,
            amount,
            category: Category::from_name(category),
        })
    }
}

impl<S> FromRequest<S> for TaskDescriptor
where
    S: Send + Sync,
{
    type Rejection = CreateTaskError;

    async fn from_request(req: axum::extract::Request, _: &S) -> Result<Self, Self::Rejection> {
        fn get_images_buf(source: Bytes, mime: &str) -> Result<Vec<Vec<u8>>, CreateTaskError> {
            if mime.starts_with("image/") {
                return Ok(vec![source.to_vec()]);
            } else if !mime.starts_with("application/") {
                return Err(CreateTaskError::UnspecificContentType(mime.into()));
            }
            let mut bufs = Vec::new();
            let type_name = mime.split_once('/').unwrap().1;
            match type_name {
                "zip" | "zip-compressed" => {
                    let mut archive = ZipArchive::new(Cursor::new(source))?;
                    for i in 0..archive.len() {
                        let item = archive.by_index(i)?;
                        if item.is_file() {
                            bufs.push(item.bytes().collect::<Result<Vec<_>, _>>()?);
                        } else {
                            return Err(ZipError::InvalidArchive(Cow::Owned(
                                "accept files only, got dir / symlink".into(),
                            ))
                            .into());
                        }
                    }
                }
                _ => return Err(CreateTaskError::UnsupportedFileType(mime.into())),
            }
            Ok(bufs)
        }

        let content_type = UTF_8
            .decode(req.headers().get("Content-Type").unwrap().as_bytes())
            .0;
        log::debug!("receiving {}", content_type);

        let mut images_buf = None;
        let (mut lm_sampling, mut vlm_sampling) = (None, None);
        if content_type.starts_with("multipart/form-data") {
            let mut form: Multipart = req.extract().await?;
            while let Some(field) = form.next_field().await? {
                let name = field.name().unwrap().to_string();
                match name.as_str() {
                    "image" => {
                        let mime = field
                            .content_type()
                            .ok_or(CreateTaskError::UnspecificContentType("image".to_string()))?
                            .to_string();
                        images_buf = Some(get_images_buf(field.bytes().await?, &mime)?);
                    }
                    "lm_sampling" | "vlm_sampling" => {
                        if let Some(mime) = field.content_type()
                            && mime != "application/json"
                        {
                            return Err(CreateTaskError::InvalidField(name.to_string()));
                        }
                        let value: SimpleSamplingParams =
                            serde_json::from_str(field.text().await?.as_str())?;
                        if name.starts_with("lm") {
                            lm_sampling = Some(value)
                        } else {
                            vlm_sampling = Some(value)
                        }
                    }
                    _ => {
                        return Err(CreateTaskError::UnknownField(name.to_string()));
                    }
                }
            }
        } else {
            let mime = content_type.to_string();
            let buf: Bytes = req.extract().await?;
            images_buf = Some(get_images_buf(buf, &mime)?);
        }
        if images_buf.is_none() {
            return Err(CreateTaskError::MissingField("image".to_string()));
        }

        Ok(Self {
            images_buf: images_buf.unwrap(),
            lm_sampling,
            vlm_sampling,
        })
    }
}

#[derive(Debug, Clone)]
pub struct TaskControlBlock {
    id: String,
    state: Arc<RwLock<State>>,
}

impl TaskControlBlock {
    pub fn new() -> Self {
        Self {
            id: key::generate_random_key(),
            state: Arc::new(RwLock::new(Default::default())),
        }
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn state(&self) -> State {
        self.state.read().unwrap().clone()
    }

    pub fn set_state(&self, state: State) {
        *self.state.write().unwrap() = state
    }
}

#[derive(Debug, Clone, Display, Default)]
pub enum State {
    #[strum(to_string = "pending")]
    #[default]
    Pending,
    #[strum(to_string = "running")]
    Running,
    #[strum(to_string = "finished")]
    Finished(Result<Success, Arc<RunTaskError>>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Success(pub Bill);

impl Serialize for State {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(self.to_string().as_str())
    }
}

impl<'de> Deserialize<'de> for State {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        match s.as_str() {
            "pending" => Ok(State::Pending),
            "running" => Ok(State::Running),
            "finished" => Ok(State::Finished(Err(Arc::new(RunTaskError::Generic(
                anyhow::anyhow!("deserialized finished state without result"),
            ))))),
            _ => Err(serde::de::Error::custom(format!("unknown state: {}", s))),
        }
    }
}

impl Serialize for TaskControlBlock {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let state = self.state.read().unwrap().clone();
        let result = match &state {
            State::Finished(result) => Some(result),
            _ => None,
        };
        let mut sstate = serializer.serialize_struct("Task", result.map(|_| 4).unwrap_or(2))?;
        sstate.serialize_field("id", &self.id)?;
        sstate.serialize_field("state", &state)?;
        if let Some(result) = result {
            sstate.serialize_field("success", &result.as_ref().ok().clone())?;
            sstate.serialize_field(
                "error",
                &result.as_ref().err().map(|err| err.to_string()).clone(),
            )?;
        }
        sstate.end()
    }
}

impl<'de> Deserialize<'de> for TaskControlBlock {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct TaskData {
            id: String,
            state: String,
            success: Option<Success>,
            error: Option<String>,
        }

        let data = TaskData::deserialize(deserializer)?;
        let state = match data.state.as_str() {
            "pending" => State::Pending,
            "running" => State::Running,
            "finished" => {
                if let Some(success) = data.success {
                    State::Finished(Ok(success))
                } else if let Some(error) = data.error {
                    State::Finished(Err(Arc::new(RunTaskError::Generic(anyhow::anyhow!(error)))))
                } else {
                    return Err(serde::de::Error::custom(
                        "finished state without success or error",
                    ));
                }
            }
            _ => {
                return Err(serde::de::Error::custom(format!(
                    "unknown state: {}",
                    data.state
                )));
            }
        };
        Ok(TaskControlBlock {
            id: data.id,
            state: Arc::new(RwLock::new(state)),
        })
    }
}

#[cfg(test)]
mod tests {
    use std::{path::PathBuf, str::FromStr};
    use tokio::fs;

    use crate::schedule;

    use super::*;

    #[tokio::test]
    async fn test_lm() {
        _ = pretty_env_logger::try_init();
        Category::load_from_names(["Shopping", "Food", "Transport", "Rent"]);

        let lm = schedule::default_lm_model().await.unwrap();
        log::debug!("model building finished");
        let description = r#"- In the top bar, there is no header text, indicating this image is a screenshot of a social media post, telling the user an inco
ming purchase of a vivo X200promini phone
- For main content, there are several items, including six photos of a gray vivo phone from different angles, a text block with hashtags and product details, an
d a post count
- For bottom section, there is a comment section with a "Leave a Comment" button, indicating the user can interact with the post
- The purchase is originally 2188, and is discounted by 0 bringing the final amount to 2188"#;
        let req = TaskDescriptor {
            images_buf: Vec::new(),
            lm_sampling: None,
            vlm_sampling: None,
        };
        let bill = req.run_with_description(&lm, description).unwrap();
        log::info!("{:?}", bill);
    }

    #[tokio::test]
    async fn test_vlm() {
        _ = pretty_env_logger::try_init();
        let screenshot_path = PathBuf::from_str(env!("CARGO_MANIFEST_DIR"))
            .unwrap()
            .join("asset/second-hand-horse-screenshot.jpeg");
        let screenshot_content = fs::read(screenshot_path).await.unwrap();
        let task = TaskDescriptor {
            images_buf: vec![screenshot_content],
            ..Default::default()
        };
        let vlm = schedule::default_vlm_model().await.unwrap();
        log::debug!("model building finished");
        let description = task.get_description(&vlm).unwrap();
        log::info!(target: "model output", "{}", description);
    }
}
