use std::sync::{Arc, LazyLock, Mutex};

use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct Bill {
    pub notes: String,
    pub amount: f32,
    pub category: Option<Category>,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct Category(usize);

impl Category {
    pub fn name(&self) -> String {
        CATEGORIES.lock().unwrap().as_ref().unwrap()[self.0].clone()
    }

    pub fn all_cases() -> Vec<Category> {
        Vec::from_iter(
            (0..CATEGORIES.lock().unwrap().as_ref().unwrap().len()).map(|idx| Category(idx)),
        )
    }

    pub fn from_name(name: impl AsRef<str>) -> Option<Category> {
        CATEGORIES
            .lock()
            .unwrap()
            .as_ref()
            .unwrap()
            .iter()
            .position(|n| n == name.as_ref())
            .map(|idx| Category(idx))
    }

    pub fn load_from_names<Iter>(iter: Iter)
    where
        Iter: IntoIterator,
        Iter::Item: AsRef<str>,
    {
        let names = Vec::from_iter(iter.into_iter().map(|name| name.as_ref().to_string()));
        *CATEGORIES.lock().unwrap() = Some(names);
    }
}

static CATEGORIES: LazyLock<Arc<Mutex<Option<Vec<String>>>>> =
    LazyLock::new(|| Arc::new(Mutex::new(None)));
