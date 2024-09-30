FROM --platform=linux/amd64 python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt

# Install Streamlit explicitly
RUN pip install --no-cache-dir streamlit==1.22.0

# Expose the port that Streamlit uses (default is 8501)
EXPOSE 80

# Run the Streamlit app
CMD ["sh", "-c", "streamlit run streamlit.py --server.port 80 --server.address 0.0.0.0"]