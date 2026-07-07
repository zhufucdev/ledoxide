use std::sync::{Arc, RwLock};

use serde::{Deserialize, Serialize, ser::SerializeStruct};
use smol_str::SmolStr;
use strum::Display;

use crate::{bill::Bill, error::RunTaskError, key};

pub trait TaskDescriptor {
    fn images(&self) -> Vec<&[u8]>;
    fn category_names(&self) -> Vec<SmolStr>;
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
            "finished" => Ok(State::Finished(Err(Arc::new(RunTaskError::Runner(
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
                    State::Finished(Err(Arc::new(RunTaskError::Runner(anyhow::anyhow!(error)))))
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
