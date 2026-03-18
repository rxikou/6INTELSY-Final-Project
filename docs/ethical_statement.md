# Ethical Statement and Risk Register

## Overview
As automated systems are increasingly deployed to flag misinformation, it is critical to ensure these tools do not inadvertently suppress legitimate speech, amplify existing biases, or compromise user privacy. This document outlines the primary ethical risks identified for this project and the active mitigations implemented by the team.

## Risk 1: Algorithmic Bias and Underrepresentation
* **The Risk:** Machine learning models inherently learn the biases present in their training data. If the dataset disproportionately labels statements from specific demographics, political affiliations, or dialects as "Fake," the model will learn to penalize those groups unfairly.
* **Mitigation:** The team conducted rigorous Exploratory Data Analysis (EDA) on the LIAR dataset to verify class balance and sequence lengths. By utilizing the Macro F1-score rather than raw accuracy, the evaluation pipeline actively penalizes the model if it ignores minority classes. Future iterations may include slice analysis across available metadata to ensure equitable performance.

## Risk 2: Over-Reliance and False Positives (Censorship)
* **The Risk:** A high false-positive rate (flagging true statements as misinformation) can lead to the unjust suppression of valid public discourse. Users or platforms relying entirely on a binary output might censor accurate information.
* **Mitigation:** The architecture incorporates a Reinforcement Learning (RL) agent specifically designed to optimize the decision boundary. Instead of relying on a rigid default threshold (0.5), the RL agent dynamically adjusts the cutoff based on validation feedback to balance precision and recall. Furthermore, the model is designed to be a supportive tool requiring human review for low-confidence predictions rather than an autonomous censor.

## Risk 3: Data Privacy and Provenance
* **The Risk:** Utilizing web-scraped text data carries the risk of inadvertently capturing and memorizing Personally Identifiable Information (PII) from private individuals.
* **Mitigation:** The project strictly utilizes the LIAR dataset, which consists exclusively of public statements made by public figures in broadcast and print media, inherently bypassing private consent issues. Additionally, the reproducible data pipeline (`get_data.py`) ensures that raw data is downloaded directly from the academic source and processed locally, keeping large datasets out of public version control.