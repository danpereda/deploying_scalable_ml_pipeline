The main purpose of this repo is to deploy an scalable ML pipeline following best practices.
The app can be seen at: [heroku](https://deploying-scalable-ml-pipeline.herokuapp.com/)

# Repo Structure

```bash
.
├── data
│   ├── census.csv
│   ├── X_train.txt
│   └── y_train.txt
├── dvc_on_heroku_instructions.md
├── Intructions.md
├── img
│   ├── metric_on_slices_1.png
│   └── metric_on_slices_2.png
├── main.py
├── model
│   ├── encoder.pkl
│   ├── lb.pkl
│   └── model.pkl
├── model_card_template.md
├── notebooks
│   ├── eda_model.ipynb
│   ├── requests_testing.ipynb
│   └── slice_output.txt
├── Procfile
├── README.md
├── requirements.txt
├── runtime.txt
├── sanitycheck.py
├── screenshots
│   ├── continuos_integration.png
│   ├── example.png
│   ├── live_get.png
│   └── live_post.png
├── setup.py
├── starter
│   ├── __init__.py
│   ├── ml
│   │   ├── data.py
│   │   ├── __init__.py
│   │   ├── model.py
│   ├── starter
│   │   ├── ml
│   └── train_model.py
├── test_app.py
└── test_model.py
```
