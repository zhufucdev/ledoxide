use clap::Parser;

#[derive(Debug, Parser)]
#[command(version, about, long_about = None)]
/// Client pulling based HTTP server to implement a VLM based bookkeeping workflow.
pub struct Args {
    #[arg(short, long, default_value = "127.0.0.1:3100")]
    pub bind: String,
    #[arg(short, long)]
    pub auth_key: Option<String>,
    #[arg(
        short, long,
        default_values_t = ["Gorceries".to_string(), "Transport".to_string(), "Rent".to_string(), "Entertainment".to_string(), "Shopping".to_string(), "Drink".to_string(), "Food".to_string(), "Drink".to_string()])]
    pub categories: Vec<String>,
}
