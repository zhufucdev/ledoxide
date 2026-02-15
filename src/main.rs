use axum::{
    Json,
    extract::{Path, State},
    routing::{get, post},
};
use clap::Parser;
use serde::{Deserialize, Serialize};
use tokio::net::TcpListener;

use crate::{
    error::GetTaskError,
    key::ValidKey,
    state::AppState,
    task::{TaskControlBlock, TaskDescriptor},
};

mod bill;
mod cli;
mod error;
mod key;
mod models;
mod schedule;
mod state;
mod task;

#[tokio::main(flavor = "multi_thread")]
async fn main() {
    pretty_env_logger::init();
    let args = cli::Args::parse();
    let auth_key = match args.auth_key {
        Some(key) => key,
        None => {
            let random_key = key::generate_random_key();
            log::error!("missing authorization key, using a random one: {random_key}");
            random_key
        }
    };

    let app = axum::Router::new()
        .route("/", get(index))
        .route("/create_task", post(create_task))
        .route("/get_task/{task_id}", get(get_task))
        .with_state(AppState::new(auth_key));
    let listener = TcpListener::bind(args.bind).await.expect("Failed to bind");
    log::info!("Listening on http://{}", listener.local_addr().unwrap());
    axum::serve(listener, app).await.unwrap();
}

async fn index() -> &'static str {
    return concat!(env!("CARGO_PKG_NAME"), " ", env!("CARGO_PKG_VERSION"));
}

#[axum::debug_handler]
async fn create_task(
    _: ValidKey,
    state: State<AppState>,
    task: TaskDescriptor,
) -> Json<TaskControlBlock> {
    Json(state.scheduler().create_task(task).await)
}

async fn get_task(
    _: ValidKey,
    state: State<AppState>,
    Path(GetTaskParams { task_id }): Path<GetTaskParams>,
) -> Result<Json<TaskControlBlock>, GetTaskError> {
    state
        .scheduler()
        .get_task(task_id)
        .await
        .ok_or(GetTaskError::NotFound)
        .map(Json)
}

#[derive(Debug, Deserialize, Serialize)]
struct GetTaskParams {
    task_id: String,
}
