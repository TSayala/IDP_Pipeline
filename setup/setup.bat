@echo off
REM Create a virtual environment
python -m venv documentProcessor

REM Activate the virtual environment
call documentProcessor\Scripts\activate

REM Install dependencies
pip install -r dependencies.txt

REM List installed packages
pip list