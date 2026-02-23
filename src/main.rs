use axum::{
    Json,
    extract::{Path, State},
    routing::{get, post},
};
use clap::Parser;
use serde::{Deserialize, Serialize};
use tokio::net::TcpListener;

use crate::{
    bill::Category,
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
    Category::load_from_names(args.categories);
    let auth_key = match args.auth_key {
        Some(key) => key,
        None => match std::env::var("AUTH_KEY") {
            Ok(key) => key,
            Err(_) => {
                let random_key = key::generate_random_key();
                log::error!("missing authorization key, using a random one: {random_key}");
                random_key
            }
        },
    };

    let app = app(auth_key);
    let listener = TcpListener::bind(args.bind).await.expect("failed to bind");
    log::info!("Listening on http://{}", listener.local_addr().unwrap());
    axum::serve(listener, app).await.unwrap();
}

fn app(auth_key: impl ToString) -> axum::Router {
    axum::Router::new()
        .route("/", get(index))
        .route("/create_task", post(create_task))
        .route("/get_task/{task_id}", get(get_task))
        .with_state(AppState::new(auth_key.to_string()))
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
        .await?
        .ok_or(GetTaskError::NotFound)
        .map(Json)
}

#[derive(Debug, Deserialize, Serialize)]
struct GetTaskParams {
    task_id: String,
}

#[cfg(test)]
mod tests {
    use std::{path::PathBuf, str::FromStr, time::Duration};

    use axum::{body::Body, extract::Request};
    use futures::{StreamExt, TryFutureExt, TryStreamExt};
    use reqwest::{StatusCode, multipart::Form};
    use tower::{Service, util::ServiceExt};

    use crate::bill::Category;

    use super::*;

    #[tokio::test]
    async fn test_worflow() {
        pretty_env_logger::init();
        let auth_key = "WK1wJ5ipiVvSdmdCPqNx8up8qj8GCwbm_";
        Category::load_from_names(["Shopping", "Food", "Transport", "Rent"]);
        fn check_finished_state(success: task::Success) {
            let bill = success.0;
            assert_eq!(bill.amount, 2188f32);
            assert_eq!(
                bill.category,
                Some(Category::from_name("Shopping").unwrap())
            )
        }

        let mut app = app(auth_key).into_service();
        let screenshot_path = PathBuf::from_str(env!("CARGO_MANIFEST_DIR"))
            .unwrap()
            .join("asset/second-hand-horse-screenshot.jpeg");
        let form = Form::new()
            .file("image", screenshot_path.to_str().unwrap())
            .await
            .unwrap();
        let request = Request::builder()
            .method("POST")
            .uri("/create_task")
            .header(
                "Content-Type",
                format!("multipart/form-data; boundary={}", form.boundary()),
            )
            .header("Authorization", format!("Bearer {}", auth_key))
            .body(Body::from_stream(form.into_stream()))
            .unwrap();
        let response = app.ready().await.unwrap().call(request).await.unwrap();
        let status = response.status();
        let body = response
            .into_body()
            .into_data_stream()
            .map_ok(|chunk| chunk.to_vec())
            .try_concat()
            .await
            .unwrap();
        assert_eq!(
            status,
            StatusCode::OK,
            "{}",
            String::from_utf8(body).unwrap()
        );
        let tcb: TaskControlBlock =
            serde_json::from_str(String::from_utf8(body).unwrap().as_str()).unwrap();
        let task_id = tcb.id().to_string();
        if let task::State::Finished(Ok(success)) = tcb.state() {
            check_finished_state(success);
            return;
        }
        loop {
            tokio::time::sleep(Duration::from_secs(1)).await;
            let request = Request::builder()
                .uri(format!("/get_task/{}", task_id))
                .header("Authorization", format!("Bearer {}", auth_key))
                .body(Body::empty())
                .unwrap();
            let response = app.ready().await.unwrap().call(request).await.unwrap();
            let status = response.status();
            let body = response
                .into_body()
                .into_data_stream()
                .map_ok(|chunk| chunk.to_vec())
                .try_concat()
                .await
                .unwrap();
            assert_eq!(
                status,
                StatusCode::OK,
                "{}",
                String::from_utf8(body).unwrap()
            );

            let tcb: TaskControlBlock =
                serde_json::from_str(String::from_utf8(body).unwrap().as_str()).unwrap();
            if let task::State::Finished(state) = tcb.state() {
                match state {
                    Ok(success) => check_finished_state(success),
                    Err(err) => log::error!("{err}"),
                }
                return;
            }
        }
    }
}
