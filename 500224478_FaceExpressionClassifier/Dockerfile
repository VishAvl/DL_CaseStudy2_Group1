# Get the image of python
FROM python:3.9

# Copy all the files from local-dir to machine dir
COPY . .

# Set the current directory as working dir
WORKDIR /

# Install the requirements
RUN pip install --no-cache-dir -r ./requirements.txt

# Launch the server
CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "7860"]