pub const RECORDS: [&str; 6] = [
    "[User]",
    "[Assistant]",
    "[Assistant reasoning]",
    "[System]",
    "[Tool call ",
    "[Tool result ",
];

const LABEL: &str = "[Assistant]";

#[derive(Default, Clone, Debug)]
pub struct TurnGuard {
    scanned: usize,
    cut: Option<&'static str>,
}

impl TurnGuard {
    pub fn cut(&self) -> Option<&'static str> {
        self.cut
    }

    pub fn feed(&mut self, text: &mut String) -> Option<&'static str> {
        if self.cut.is_some() {
            return None;
        }
        self.scanned = self.scanned.min(text.len());
        loop {
            let rest = &text[self.scanned..];
            let (line, complete) = match rest.find('\n') {
                Some(i) => (&rest[..i], true),
                None => (rest, false),
            };
            let head = line.trim_start_matches([' ', '\t']);
            if head.starts_with(LABEL) && text[..self.scanned].trim().is_empty() {
                let at = self.scanned + (line.len() - head.len());
                text.replace_range(at..at + LABEL.len(), "");
                continue;
            }
            if let Some(m) = RECORDS.iter().find(|m| head.starts_with(**m)) {
                text.truncate(self.scanned);
                self.cut = Some(*m);
                return self.cut;
            }
            if !complete {
                return None;
            }
            self.scanned += line.len() + 1;
        }
    }
}

pub fn cut_transcript(text: &str) -> (String, Option<&'static str>) {
    let mut out = text.to_string();
    let cut = TurnGuard::default().feed(&mut out);
    (out, cut)
}

pub fn strip_reasoning(text: &str) -> String {
    let mut out = text.to_string();
    for (open, close) in [("<think>", "</think>"), ("<thinking>", "</thinking>")] {
        while let Some(a) = out.find(open) {
            let Some(rel) = out[a + open.len()..].find(close) else {
                break;
            };
            let b = a + open.len() + rel + close.len();
            out.replace_range(a..b, "");
        }
    }
    for tag in [
        "<think>",
        "</think>",
        "<thinking>",
        "</thinking>",
        "<response>",
        "</response>",
    ] {
        out = out.replace(tag, "");
    }
    out.replace("thinkresponse", "\n\n")
}

pub fn marker_name(m: &str) -> String {
    m.trim_matches(|c: char| c == '[' || c == ']' || c.is_whitespace())
        .replace(' ', "_")
}
