[![maintained by lamazabuelg](https://img.shields.io/badge/maintained%20by-Luis%20%C3%81ngel%20Mazabuel%20Garc%C3%ADa-yellowgreen)](https://img.shields.io/badge/maintained%20by-Luis%20%C3%81ngel%20Mazabuel%20Garc%C3%ADa-yellowgreen)
[![PythonVersion](https://img.shields.io/pypi/pyversions/gino_admin)](https://img.shields.io/pypi/pyversions/gino_admin)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)
# Machine Learning Challenge of ACID Labs

This is a submition for the Machine Learning Engineer Challenge purposed by ACID Labs on march, 2023.

## Usage

### API DEMO

For better understandability of the servire the root path in the repository has a **ACID-mle-challenge_Demo_API_usage.ipynb** file where you can find an end-to-end test for several endpoints simulating the real use of the service. From managing files to configuring, developing, using and evaluating Data Science models.
![image](https://user-images.githubusercontent.com/69969517/226226889-e2d0e522-0722-4195-81d3-cc090ed932c1.png)


### FASTAPI - SWAGGER UI

For use the SWAGGER UI for the API just open this link in your browser: https://acid-mle-challenge-3skroqvvwa-ue.a.run.app/docs .
![image](https://user-images.githubusercontent.com/69969517/226226985-f8b89565-d0e0-48d3-b22a-95094f0d15cb.png)

### PYTHON USAGE

```python
import requests
URL = "https://acid-mle-challenge-3skroqvvwa-ue.a.run.app/docs"
response = requests.get(f"{URL}files/all_files")
rresponse.content
# b'[{"input":[]},{"output":[]}]'
```

## For Local launch

### Virtual Environment
I suggest to use a **virtual environment**. Use the package manager [pip](https://pip.pypa.io/en/stable/) to install **virtualenv**:

```bash
pip install virtualenv
```

Once **virtualenv** library is installed, execute:

```bash
python3.9 -m virtualenv venv
```

Activate your **virtual environment** before continue:

```bash
source venv/Scripts/activate
```

### Requirements

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install python requirements for the project.

```bash
pip install -r requirements.txt
```

After that you will be able to execute the **main.py** file and test it locally.

## Contributing
As a private Challenge, Pull Request are welcome but just for possible improvements, comments and suggestions.

## License

[MIT](https://choosealicense.com/licenses/mit/)
