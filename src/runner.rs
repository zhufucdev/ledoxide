use std::{cell::RefCell, io::IsTerminal, num::NonZeroU32, rc::Rc, sync::LazyLock};

use async_stream::try_stream;
use encoding_rs::UTF_8;
use futures::{Stream, TryStreamExt};
use hf_hub::api::tokio::ApiBuilder;
use llama_cpp_2::{
    LlamaContextLoadError,
    context::{LlamaContext, params::LlamaContextParams},
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{LlamaChatMessage, LlamaChatTemplate, LlamaModel},
    mtmd::{self, MtmdBitmap, MtmdContext, MtmdInputText},
};
use strum::Display;
use tokio::pin;

use crate::{
    error::{CreateLlamaCppRunnerError, RunnerError},
    sample::{LlguidanceSamplingParams, SimpleSamplingParams},
};

#[macro_export]
macro_rules! StreamableRunnerResponse {
    ($key:ident) => {
        $key Stream<Item = Result<String, RunnerError>>
    };
}

pub trait TextLmRunner {
    fn stream_lm_response(&self, request: TextLmRequest) -> StreamableRunnerResponse!(impl);
}

pub trait VisionLmRunner {
    fn stream_vlm_response(&self, request: VisionLmRequest) -> StreamableRunnerResponse!(impl);
}

pub struct RunnerRequest<M> {
    pub messages: Vec<(MessageRole, M)>,
    pub sampling: SimpleSamplingParams,
    pub llguidance: Option<LlguidanceSamplingParams>,
    pub max_seq: i32,
}

impl<M> Default for RunnerRequest<M> {
    fn default() -> Self {
        Self {
            messages: vec![],
            sampling: Default::default(),
            llguidance: None,
            max_seq: i32::MAX,
        }
    }
}

pub type TextLmRequest = RunnerRequest<String>;
pub type VisionLmRequest = RunnerRequest<ImageOrText>;

async fn get_whole_response(
    stream: StreamableRunnerResponse!(impl),
) -> Result<String, RunnerError> {
    let mut response = String::new();
    pin!(stream);
    while let Some(chunk) = stream.try_next().await? {
        response += chunk.as_str();
    }
    Ok(response)
}

pub trait TextLmRunnerExt {
    async fn get_lm_response(&self, request: TextLmRequest) -> Result<String, RunnerError>;
}

pub trait VisionLmRunnerExt {
    async fn get_vlm_response(&self, request: VisionLmRequest) -> Result<String, RunnerError>;
}

impl<T> TextLmRunnerExt for T
where
    T: TextLmRunner,
{
    async fn get_lm_response(&self, request: TextLmRequest) -> Result<String, RunnerError> {
        Ok(get_whole_response(self.stream_lm_response(request)).await?)
    }
}

impl<T> VisionLmRunnerExt for T
where
    T: VisionLmRunner,
{
    async fn get_vlm_response(&self, request: VisionLmRequest) -> Result<String, RunnerError> {
        Ok(get_whole_response(self.stream_vlm_response(request)).await?)
    }
}

#[derive(Debug, Clone, Display, PartialEq, Eq)]
pub enum MessageRole {
    #[strum(to_string = "assistant")]
    Assistant,
    #[strum(to_string = "user")]
    User,
    #[strum(to_string = "system")]
    System,
}

pub enum ImageOrText {
    Text(String),
    Image(image::DynamicImage),
}

pub struct Gemma4bRunner {
    model: LlamaModel,
    chat_template: LlamaChatTemplate,
    mtmd_ctx: MtmdContext,
    ctx_size: NonZeroU32,
}

static LLAMA_BACKEND: LazyLock<LlamaBackend> = LazyLock::new(|| LlamaBackend::init().unwrap());

impl Gemma4bRunner {
    pub async fn new() -> Result<Self, CreateLlamaCppRunnerError> {
        let repo = ApiBuilder::new()
            .with_progress(std::io::stdin().is_terminal())
            .build()?
            .model("google/gemma-3-4b-it-qat-q4_0-gguf".to_string());
        let model = LlamaModel::load_from_file(
            &LLAMA_BACKEND,
            repo.get("gemma-3-4b-it-q4_0.gguf").await?,
            &Default::default(),
        )?;

        let mtmd_ctx = MtmdContext::init_from_file(
            repo.get("mmproj-model-f16-4B.gguf")
                .await?
                .to_str()
                .unwrap(),
            &model,
            &Default::default(),
        )?;

        let chat_template = model.chat_template(None)?;

        Ok(Self {
            model,
            mtmd_ctx,
            chat_template,
            ctx_size: 10240u32.try_into().unwrap(),
        })
    }
}

impl Gemma4bRunner {
    fn generate_response(
        &self,
        ctx: &RefCell<LlamaContext<'_>>,
        request: &VisionLmRequest,
    ) -> Result<impl Iterator<Item = Result<String, RunnerError>>, RunnerError> {
        // Preprocess the message, flattening media
        let media_marker = mtmd::mtmd_default_marker();
        let messages = request
            .messages
            .iter()
            .fold(
                Vec::<(MessageRole, String)>::new(),
                |mut acc, (role, message)| {
                    let text = match message {
                        ImageOrText::Text(text) => text,
                        ImageOrText::Image(_) => media_marker,
                    };
                    if let Some(last) = acc.last()
                        && last.0 == *role
                    {
                        // merge adjacent
                        acc.remove(acc.len() - 1);
                        acc.push((role.clone(), format!("{0}\n{text}", acc[acc.len() - 1].1)));
                        acc
                    } else {
                        acc.push((role.clone(), text.to_string()));
                        acc
                    }
                },
            )
            .into_iter()
            .map(|(role, content)| LlamaChatMessage::new(role.to_string(), content))
            .collect::<Result<Vec<_>, _>>()
            .expect("message preprocessing failed");
        log::debug!(target: "gemma", "preprocessed messages: {messages:?}");

        // Aggregate images
        let chat_template = self.model.chat_template(None)?;
        let formatted_prompt = self
            .model
            .apply_chat_template(&chat_template, &messages, true)?;
        let bitmaps = request
            .messages
            .iter()
            .filter_map(|msg| match &msg.1 {
                ImageOrText::Image(image) => Some(image),
                _ => None,
            })
            .enumerate()
            .map(|(idx, im)| {
                MtmdBitmap::from_image_data(
                    im.width(),
                    im.height(),
                    im.to_rgb8().to_vec().as_slice(),
                )
                .expect(format!("image#{} has corrupted RGB data", idx).as_str())
            })
            .collect::<Vec<_>>();
        let bitmap_refs = bitmaps.iter().collect::<Vec<_>>();
        let chunks = self.mtmd_ctx.tokenize(
            MtmdInputText {
                text: formatted_prompt,
                add_special: true,
                parse_special: true,
            },
            &bitmap_refs,
        )?;
        log::debug!(target: "gemma", "tokenization resulted in {} chunks", chunks.len());
        let n_past = chunks.eval_chunks(&self.mtmd_ctx, &ctx.borrow(), 0, 0, 1, true)?;

        // Generate response
        let sampler = RefCell::new(request.sampling.to_llama());
        let decoder = RefCell::new(UTF_8.new_decoder());
        let batch = RefCell::new(LlamaBatch::new(self.ctx_size.get() as usize, 1));
        let ctx = Rc::new(ctx);
        let (ctx_1, ctx_2) = (ctx.clone(), ctx.clone());
        let iter = (0..request.max_seq)
            .map(move |step| {
                let token = (&sampler.borrow_mut().sample(&ctx_1.borrow(), -1)).clone();
                sampler.borrow_mut().accept(token);
                (step, token)
            })
            .take_while(|(_, token)| self.model.is_eog_token(token.clone()))
            .map(move |(step, token)| -> Result<String, RunnerError> {
                let mut batch = batch.borrow_mut();
                batch.clear();
                batch.add(token, n_past + step, &[0], true)?;
                ctx_2.borrow_mut().decode(&mut batch)?;
                let piece =
                    self.model
                        .token_to_piece(token, &mut decoder.borrow_mut(), true, None)?;
                Ok(piece)
            });
        Ok(iter)
    }

    fn new_context_window(&self) -> Result<LlamaContext<'_>, LlamaContextLoadError> {
        self.model.new_context(
            &LLAMA_BACKEND,
            LlamaContextParams::default().with_n_ctx(Some(self.ctx_size)),
        )
    }
}

impl TextLmRunner for Gemma4bRunner {
    fn stream_lm_response(&self, request: TextLmRequest) -> StreamableRunnerResponse!(impl) {
        try_stream! {
            let ctx = RefCell::new(self.new_context_window()?);
            let vlm_req: VisionLmRequest = request.into();
            for chunk in self.generate_response(&ctx, &vlm_req)? {
                yield chunk?;
            }
        }
    }
}

impl VisionLmRunner for Gemma4bRunner {
    fn stream_vlm_response(&self, request: VisionLmRequest) -> StreamableRunnerResponse!(impl) {
        try_stream! {
            let ctx = RefCell::new(self.new_context_window()?);
            for chunk in self.generate_response(&ctx, &request)? {
                yield chunk?;
            }
        }
    }
}

impl From<TextLmRequest> for VisionLmRequest {
    fn from(value: TextLmRequest) -> Self {
        Self {
            messages: value
                .messages
                .into_iter()
                .map(|(role, text)| (role, ImageOrText::Text(text)))
                .collect(),
            sampling: value.sampling,
            llguidance: value.llguidance,
            max_seq: value.max_seq,
        }
    }
}
