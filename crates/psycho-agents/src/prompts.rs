//! Prompt templates for agents.

/// System prompt for Interpreter Agent
pub const INTERPRETER_SYSTEM_PROMPT: &str = r#"You are a behavioral psychology expert specializing in nonverbal communication and movement analysis. Your role is to interpret Laban Movement Analysis (LMA) scores and Big Five (OCEAN) personality traits to generate actionable behavioral insights.

You will receive:
1. LMA Effort scores (Space, Time, Weight, Flow)
2. OCEAN personality scores (Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism)
3. Hesitation metrics and movement patterns

Your task is to provide a concise behavioral summary that:
- Identifies the customer's emotional state
- Notes key behavioral patterns
- Highlights needs or concerns
- Suggests general engagement approach

Be professional, empathetic, and focused on actionable insights. Keep your response under 150 words."#;

/// System prompt for Strategist Agent
pub const STRATEGIST_SYSTEM_PROMPT: &str = r#"You are an expert automotive sales strategist. Your role is to synthesize behavioral insights with customer preferences to create a tailored sales strategy.

You will receive:
1. Behavioral summary from the Interpreter
2. Customer questionnaire data (if available)
3. Sales context (vehicle interest, budget, priorities)

Your task is to provide:
- Sales approach (e.g., Consultative, Data-driven, Relationship-focused)
- Key themes to emphasize (e.g., Safety, Performance, Value, Innovation)
- Specific features/benefits to highlight
- Potential objections to address proactively
- Communication style recommendations

Be strategic, specific, and actionable. Keep your response structured and under 200 words."#;

/// System prompt for Copywriter Agent with Chain-of-Thought
pub const COPYWRITER_SYSTEM_PROMPT: &str = r#"You are an expert copywriter specializing in personalized sales conversations. Your role is to generate three opening conversation hooks based on the sales strategy.

You will receive:
1. Sales strategy from the Strategist
2. Customer profile summary
3. Specific themes to emphasize

Use Chain-of-Thought reasoning to:
Step 1: Analyze the customer's primary need/concern
Step 2: Identify the emotional appeal that resonates
Step 3: Choose an opening angle (question, statement, or observation)
Step 4: Craft the hook with specific details

Generate exactly THREE distinct opening hooks:
- Each should be 2-3 sentences
- Conversational and natural tone
- Incorporate the recommended themes
- Create immediate relevance and engagement

Format each hook as:
Hook 1: [text]
Hook 2: [text]
Hook 3: [text]"#;

/// Template for Interpreter Agent input
pub fn format_interpreter_input(
    ocean_summary: &str,
    lma_summary: &str,
    hesitation_summary: &str,
    movement_patterns: &str,
) -> String {
    format!(
        r#"=== CUSTOMER MOVEMENT ANALYSIS ===

OCEAN Personality Profile:
{}

LMA Effort Qualities:
{}

Hesitation Patterns:
{}

Movement Characteristics:
{}

Please provide a behavioral interpretation and engagement recommendation."#,
        ocean_summary, lma_summary, hesitation_summary, movement_patterns
    )
}

/// Template for Strategist Agent input
pub fn format_strategist_input(
    behavioral_summary: &str,
    questionnaire_data: Option<&str>,
    context: &str,
) -> String {
    let questionnaire = questionnaire_data.unwrap_or("No questionnaire data available.");

    format!(
        r#"=== SALES STRATEGY DEVELOPMENT ===

Behavioral Interpretation:
{}

Customer Questionnaire:
{}

Sales Context:
{}

Please provide a comprehensive sales strategy with specific recommendations."#,
        behavioral_summary, questionnaire, context
    )
}

/// Template for Copywriter Agent input with CoT
pub fn format_copywriter_input(
    strategy: &str,
    customer_profile: &str,
    themes: &[&str],
) -> String {
    let themes_list = themes.join(", ");

    format!(
        r#"=== OPENING HOOK GENERATION ===

Sales Strategy:
{}

Customer Profile:
{}

Key Themes to Emphasize:
{}

Using Chain-of-Thought reasoning, generate three personalized opening conversation hooks."#,
        strategy, customer_profile, themes_list
    )
}

/// Parse Chain-of-Thought response
pub fn parse_cot_response(response: &str) -> Result<(Vec<String>, Vec<String>), String> {
    let mut steps = Vec::new();
    let mut hooks = Vec::new();

    for line in response.lines() {
        let trimmed = line.trim();

        if trimmed.starts_with("Step ") {
            steps.push(trimmed.to_string());
        } else if trimmed.starts_with("Hook ") {
            if let Some(colon_pos) = trimmed.find(':') {
                let hook = trimmed[colon_pos + 1..].trim().to_string();
                if !hook.is_empty() {
                    hooks.push(hook);
                }
            }
        }
    }

    if hooks.is_empty() {
        Err("No hooks found in response".to_string())
    } else {
        Ok((steps, hooks))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_interpreter_input() {
        let input = format_interpreter_input(
            "High Neuroticism",
            "Bound Flow, Indirect",
            "Frequent hesitation",
            "Small kinesphere",
        );

        assert!(input.contains("OCEAN"));
        assert!(input.contains("LMA"));
        assert!(input.contains("High Neuroticism"));
    }

    #[test]
    fn test_parse_cot_response() {
        let response = r#"
Step 1: Analyze customer need - Safety is priority
Step 2: Choose emotional appeal - Peace of mind
Hook 1: I noticed you're interested in family vehicles. Have you had a chance to see our advanced safety ratings?
Hook 2: Safety is so important. Let me show you how this model's collision avoidance system works.
Hook 3: Many families tell us the safety features give them real peace of mind. Is that something you're prioritizing?
        "#;

        let (steps, hooks) = parse_cot_response(response).unwrap();
        assert_eq!(steps.len(), 2);
        assert_eq!(hooks.len(), 3);
        assert!(hooks[0].contains("family vehicles"));
    }
}
