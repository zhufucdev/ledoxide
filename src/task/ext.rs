use std::fmt::Display;

use crate::bill::Category;

pub trait DisplayCategory {
    fn bullet_item(&self) -> impl Display;
}

impl<T> DisplayCategory for T
where
    T: Category,
{
    fn bullet_item(&self) -> impl Display {
        if let Some(desc) = self.description() {
            format!("- {}: {}", self.name(), desc)
        } else {
            format!("- {}", self.name())
        }
    }
}
