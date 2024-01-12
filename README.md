# Hydirb Task-Routed Fine-Tuned Large Language Models

## How to run

-   Install necessary dependencies

```
pip install -r requirements.txt
```

-   Download [Spider Dataset](https://yale-lily.github.io/spider) and place in the parent directory

-   Run the script with the following command

```
python run.py [MODEL: finance | medicine | SQL | llama2 | router] [DATASET: fin | med | nsql]
```

where the router is the hybrid model that uses llama2 to router the tasks to a fined-tuned model then executes the task.

## Dataset

There are 3 datasets used in this project:

-   fin: Finance-Task/Headline Dataset
    This dataset is designed for finance-related content, offering a focused environment to evaluate the finance model’s performance. Here is one example of the Headline dataset. The label for each headline is either "Yes" or "No".
-   med: Medicine-Task/RCT Dataset
    Utilized for the medical domain, this dataset comprises data from randomized controlled trials, providing a complex and relevant context for assessing the medicine model. There are 5 options for the dataset options: Conclusions, Methods, Background, Results, Objective.
-   nsql: Spider Dataset for SQL
    Chosen for SQL tasks, the Spider dataset Yu et al. 2019 is a comprehensive benchmark for SQL query processing, testing the SQL model’s ability to handle diverse queries and database structures. There is one database and SQL scheme associated with each data point. In our experiment, we didn’t enforce the exact match, which needs the generated SQL query to exactly match the label. Instead, the evaluation metric only checks the queried outcome from the database.

## Logging

Each exectution generates a log file in the `log/` folder, with the name `model_[MODEL_NAME]_dataset_[DATASET_NAME].log`.
