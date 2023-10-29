# Project Name

This project is designed to apply Natural Language Processing (NLP) techniques to analyze a dataset named "issues.csv." The primary objective is to use clustering algorithms to group similar problems together, making it easier to identify and prioritize issues that need to be addressed.

## Table of Contents

- [Introduction](#introduction)
- [Problem Statement](#problem-statement)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Testing](#testing)
- [Improvements](#improvements)
- [License](#license)

## Introduction

Provide a brief introduction to your project, its purpose, and its main features.

## Problem Statement

Explain the problem your project is solving or the goal it aims to achieve.

## Project Structure

Describe the structure of your project's repository, including folders, main files, and their purposes.

```
project-root/
│
├── app/
│ ├── api.py
│ └── Dockerfile
│ └── requirements.txt
│
├── data/
│ ├── input/
│ │ └── issues.csv
│ └── output/
│
├── modules/
│ ├── preprocessing.py
│ └── clustering.py
│
├── requirements.txt
│
└── README.md
```

## Getting Started

Explain how to set up and run your project. Include prerequisites and installation instructions.

### Prerequisites

- Python 3.10.2
- Docker

### Installation

1. Clone this repository to your local machine.
2. Build the Docker image and start the service: `docker compose up --build`
3. Wait until the message `Running on http://127.0.0.1:5000` appear.

## Usage

To use the API, you can use a tool like Postman.

Example API request:
`http://127.0.0.1:5000/topic_modeling?input_df_path=./data/input/issues.csv&model_name=bert`

- `input_df_path`: Path to the input CSV file.
- `model_name`: Choose between "bert" or "nmf" for topic modeling.

## Improvements

Share any ideas for future improvements or features you plan to add to your project.
1. Algorithm Selection: Evaluate and experiment with various topic modeling algorithms such as Latent Dirichlet Allocation (LDA), Latent Semantic Analysis (LSA), or Hierarchical Dirichlet Process (HDP) to identify the most suitable one for the dataset.
2. Hyperparameter Tuning: Conduct extensive hyperparameter tuning for the selected topic modeling algorithm to improve topic quality.
3. Optimal Number of Topics: Implement techniques for automatically determining the optimal number of topics, such as coherence scores, topic coherence, or grid search.
4. Custom Stopwords: Allow users to specify custom stopwords relevant to their domain or dataset to improve the quality of topics.
5. Advanced Preprocessing: Enhance text preprocessing techniques, including lemmatization, stemming, and handling special characters, to improve topic modeling quality.
