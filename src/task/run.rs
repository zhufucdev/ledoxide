use crate::{bill::Bill, error::RunTaskError};

#[trait_variant::make(Send)]
pub trait RunTask {
    type TaskDescriptor;
    async fn extract(&self, task: &Self::TaskDescriptor) -> Result<Bill, RunTaskError>;
}
