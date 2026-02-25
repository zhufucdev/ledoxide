use std::sync::{Arc, RwLock};

use axum::{
    RequestExt,
    extract::{FromRequest, Multipart},
};
use mistralrs::{Constraint, RequestBuilder, SamplingParams, TextMessageRole};
use regex::Regex;
use serde::{Deserialize, Serialize, ser::SerializeStruct};
use strum::Display;

use crate::{
    bill::{Bill, Category},
    error::{CreateTaskError, RunTaskError},
    key,
    models::ModelManager,
};

#[derive(Debug, Clone, Deserialize)]
pub struct TaskDescriptor {
    image_buf: Vec<u8>,
    lm_sampling: Option<SamplingParams>,
    vlm_sampling: Option<SamplingParams>,
}

impl TaskDescriptor {
    pub fn image_bytes(&self) -> &[u8] {
        &self.image_buf
    }

    pub fn lm_sampling(&self) -> Option<SamplingParams> {
        self.lm_sampling.clone()
    }

    pub fn vlm_sampling(&self) -> Option<SamplingParams> {
        self.vlm_sampling.clone()
    }

    pub async fn run(
        &self,
        model_manager: &ModelManager,
        vlm_id: impl AsRef<str>,
        lm_id: impl AsRef<str>,
    ) -> Result<Bill, RunTaskError> {
        let vlm = model_manager.get_model(vlm_id.as_ref()).await?.unwrap();
        let mut request = RequestBuilder::new().add_image_message(
            TextMessageRole::User,
            format!(include_str!("../prompt/description.md")),
            vec![image::load_from_memory(self.image_buf.as_ref())?],
        );
        if let Some(sampling) = &self.vlm_sampling {
            request = request.set_sampling(sampling.clone());
        }
        let response = vlm.send_chat_request(request).await?;
        let description = response.choices[0].message.content.clone().unwrap();
        drop(vlm);
        log::debug!(target: "task runner", "description: {}", description);

        let lm = model_manager.get_model(lm_id.as_ref()).await?.unwrap();
        let mut request = RequestBuilder::new()
            .set_constraint(Constraint::Lark(
                include_str!("../constraint/note_taking.lark").to_string(),
            ))
            .add_message(
                TextMessageRole::User,
                format!(include_str!("../prompt/note_taking.md"), description),
            );
        if let Some(sampling) = &self.lm_sampling {
            request = request.set_sampling(sampling.clone());
        }

        let response = lm.send_chat_request(request).await?;
        let notes = response.choices[0].message.content.clone().unwrap();
        log::debug!(target: "task runner", "notes: {}", notes);
        let mut request = RequestBuilder::new()
            .add_message(
                TextMessageRole::User,
                format!(
                    include_str!("../prompt/amount_extraction.md"),
                    notes, description
                ),
            )
            .set_constraint(Constraint::Lark(
                include_str!("../constraint/amount_extraction.lark").to_string(),
            ));
        if let Some(sampling) = &self.lm_sampling {
            request = request.set_sampling(sampling.clone());
        }
        let response = lm.send_chat_request(request).await?;
        log::debug!(target: "task runner", "amount: {}", response.choices[0].message.content.clone().unwrap());
        let numeric = Regex::new(r#"([0-9,]+\.?[0-9]{0,})"#).unwrap();
        let amount = numeric.captures(
            response.choices[0]
                .message
                .content
                .as_ref()
                .unwrap()
                .rsplit_once("\n")
                .unwrap()
                .1,
        );
        let amount: f32 = if let Some(amount) = amount {
            amount.get(1).unwrap().as_str().parse().map_err(|_| {
                RunTaskError::EmptyAmount(response.choices[0].message.content.clone().unwrap())
            })?
        } else {
            return Err(RunTaskError::EmptyAmount(
                response.choices[0].message.content.clone().unwrap(),
            ));
        };
        let mut request = RequestBuilder::new()
            .add_message(
                TextMessageRole::User,
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
            )
            .set_constraint(Constraint::Lark(format!(
                include_str!("../constraint/categorization.lark"),
                Category::all_cases()
                    .iter()
                    .map(|c| c.name())
                    .collect::<Vec<_>>()
                    .join("|")
            )));
        if let Some(sampling) = &self.lm_sampling {
            request = request.set_sampling(sampling.clone());
        }
        let response = lm.send_chat_request(request).await?;
        log::debug!(target: "task runner", "category: {}", response.choices[0].message.content.clone().unwrap());
        let category = response.choices[0]
            .message
            .content
            .as_ref()
            .map(|msg| msg.rsplit_once("\n").unwrap().1.split_once(" ").unwrap().1)
            .unwrap();

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
        let mut form: Multipart = req.extract().await?;
        let mut image_buf = None;
        let (mut lm_sampling, mut vlm_sampling) = (None, None);
        while let Some(field) = form.next_field().await? {
            let name = field.name().unwrap().to_string();
            match name.as_str() {
                "image" => {
                    image_buf = Some(field.bytes().await?.to_vec());
                }
                "lm_sampling" | "vlm_sampling" => {
                    if let Some(mime) = field.content_type()
                        && mime != "application/json"
                    {
                        return Err(CreateTaskError::InvalidField(name.to_string()));
                    }
                    let value: SamplingParams = serde_json::from_str(field.text().await?.as_str())?;
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
        if image_buf.is_none() {
            return Err(CreateTaskError::MissingField("image".to_string()));
        }

        Ok(Self {
            image_buf: image_buf.unwrap(),
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
    use std::{collections::HashMap, io::Write, path::PathBuf, str::FromStr, time::Duration};

    use mistralrs::{
        ChatCompletionChunkResponse, ChunkChoice, Delta, IsqType, ModelBuilder,
        PagedAttentionMetaBuilder, Response, ResponseOk, TextModelBuilder,
    };
    use tokio::fs;

    use crate::schedule;

    use super::*;

    #[tokio::test]
    async fn test_lm() {
        pretty_env_logger::init();
        let lm = schedule::default_lm_model().await.unwrap();
        log::debug!("model building finished");
        let description = r#"description: - In the top bar, there is no header text, indicating this image is a screenshot of a social media post or listing, telling the user an incoming sale of a vivo X200 Ultra smartphone.
- For main content, there are several items, including a collage of nine images showing the vivo X200 Ultra from various angles (back, side, front, and held in hand), with text overlays identifying the model and featuring the Zeiss logo, and a final image showing the phone's screen displaying a sales page with Chinese text.
- For bottom section, there is a text block in pink font indicating the item is for sale with hashtags, contact information (T.), price (21888), and a description of the phone's condition (16+512GB, suspected replacement screen with a minor gap, but all functions normal, with official warranty still valid for over a month).
- The bill is originally not specified, and is discounted by not specified, bringing the final amount to 21888."#;
        let request = RequestBuilder::new()
            .set_constraint(Constraint::Lark(
                include_str!("../constraint/note_taking.lark").to_string(),
            ))
            .add_message(
                TextMessageRole::User,
                format!(include_str!("../prompt/note_taking.md"), description),
            );
        let response = lm.send_chat_request(request).await.unwrap();
        log::info!(target: "lm", "{}", response.choices[0].message.content.clone().unwrap());
    }

    #[tokio::test]
    async fn test_vlm() {
        pretty_env_logger::init();
        let screenshot_path = PathBuf::from_str(env!("CARGO_MANIFEST_DIR"))
            .unwrap()
            .join("asset/second-hand-horse-screenshot.jpeg");
        let screenshot_content = fs::read(screenshot_path).await.unwrap();
        let request = RequestBuilder::new().add_image_message(
            TextMessageRole::User,
            format!(include_str!("../prompt/description.md")),
            vec![image::load_from_memory(&screenshot_content).unwrap()],
        );
        let vlm = schedule::default_vlm_model().await.unwrap();
        log::debug!("model building finished");
        let response = vlm.send_chat_request(request).await.unwrap();
        log::info!(target: "vlm", "{}", response.choices[0].message.content.clone().unwrap());
    }
}
