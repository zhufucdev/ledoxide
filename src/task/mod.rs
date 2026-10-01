pub mod descriptor;
mod ext;
pub mod run;

#[cfg(feature = "ollama")]
pub mod ollama;

#[cfg(feature = "openai")]
pub mod openai;

pub use descriptor::{State, Success, TaskControlBlock, TaskDescriptor};
pub use run::RunTask;
