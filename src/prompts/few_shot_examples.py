"""Few-shot examples for biomedical text simplification."""

# These are manually selected high-quality examples from the Cochrane-auto training set
# All examples show TRUE SIMPLIFICATION (simple version is shorter/clearer than complex)
# Selected for diverse reduction ratios (60-90%) and clear readability improvements

CURATED_EXAMPLES = [
    # Strong simplification (61.8% - removes technical details)
    {
        'complex': 'Resuscitation with a nasal interface may reduce the rate of intubation in the DR, but the evidence is very uncertain (RR 0.68, 95% CI 0.54 to 0.85; 5 studies, 1406 infants; very low-certainty evidence).',
        'simple': 'It may reduce the number of newborn babies who are intubated in the delivery room, but the evidence is very uncertain.',
        'operation': 'rephrase',
        'reduction': '61.8%'
    },
    
    # Moderate simplification (75% - clearer structure)
    {
        'complex': 'There was a statistically significant shorter operating time in the early group than the delayed group (MD -14.80 minutes, 95% CI -18.02 to -11.58).',
        'simple': 'The operating time was significantly shorter (by about 15 minutes) in the early group than the delayed group.',
        'operation': 'rephrase',
        'reduction': '75.0%'
    },
    
    # Moderate simplification (73.3% - simpler vocabulary)
    {
        'complex': 'The evidence is very uncertain about the effect of these supplements on gastrointestinal side effects.',
        'simple': 'It is unclear if the protein supplement causes any unwanted effects.',
        'operation': 'rephrase',
        'reduction': '73.3%'
    },
    
    # Moderate simplification (75% - removes jargon)
    {
        'complex': 'One study compared two biphasic pills and one triphasic pill, each containing levonorgestrel and ethinyl estradiol.',
        'simple': 'One study compared two types of two-phase pills with a three-phase pill.',
        'operation': 'rephrase',
        'reduction': '75.0%'
    },
    
    # Light simplification (75% - clearer phrasing)
    {
        'complex': 'Also, there was no effect of vitamin A supplementation on mortality or morbidity due to diarrhoea and respiratory tract infection.',
        'simple': 'Supplementation had no beneficial effects to reduce death or illness due to diarrhoea or pneumonia.',
        'operation': 'rephrase',
        'reduction': '75.0%'
    },
    
    # Minimal simplification (85.7% - minor rewording)
    {
        'complex': 'We included three trials (in five articles) with 385 opiate-using participants that measured outcomes at different follow-up periods in this review.',
        'simple': 'We included three studies in this review with 385 participants in total with follow-up periods of different length.',
        'operation': 'rephrase',
        'reduction': '85.7%'
    },
    
    # Light simplification (80% - removes redundancy)
    {
        'complex': 'Seven studies (462 eyes, 434 participants) used the Ex-PRESS, and one study (527 eyes, 527 participants) used the PreserFlo MicroShunt.',
        'simple': 'Seven studies used the Ex-PRESS (434 participants), and one study used the PreserFlo MicroShunt (527 participants).',
        'operation': 'rephrase',
        'reduction': '80.0%'
    },
    
    # Moderate simplification (75% - simpler terms)
    {
        'complex': 'We included 25 trials with 2505 participants randomised to the different pharmacological agents and inactive controls.',
        'simple': 'We identified 25 randomised clinical trials involving 2505 people undergoing laparoscopic cholecystectomy.',
        'operation': 'rephrase',
        'reduction': '75.0%'
    },
    
    # Strong simplification (60% - major condensation)
    {
        'complex': 'One of the three studies included participants with \'neurodegenerative diseases\', with MS people being a subset of the randomised population.',
        'simple': 'The third study (Ne-PAL) included participants with MS and other neurodegenerative diseases.',
        'operation': 'rephrase',
        'reduction': '60.0%'
    },
    
    # Moderate simplification (70% - cleaner structure)
    {
        'complex': 'Pooled analysis of three homogenous trials showed that needle aspiration did not significantly increase the proportion of patients with fever resolution (RR 0.60, 95% confidence interval (CI) 0.22 to 1.61).',
        'simple': 'Pooled analysis of three homogenous trials showed that needle aspiration did not significantly increase the proportion of patients with fever resolution.',
        'operation': 'rephrase',
        'reduction': '70.0%'
    }
]


def get_curated_examples(num_examples: int = 3):
    """
    Get manually curated high-quality simplification examples.
    
    All examples demonstrate TRUE simplification where the simple version is:
    - Shorter or equal length to the complex version
    - Uses clearer, more accessible language
    - Removes technical jargon where possible
    - Maintains factual accuracy
    
    Args:
        num_examples: Number of examples to return (default: 3, max: 10)
    
    Returns:
        List of example dictionaries with 'complex' and 'simple' keys
    """
    # Return only the complex/simple pairs (remove metadata)
    examples = [{'complex': ex['complex'], 'simple': ex['simple']} 
                for ex in CURATED_EXAMPLES[:num_examples]]
    return examples
