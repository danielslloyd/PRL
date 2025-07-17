# Use official Miniconda image as base
FROM continuumio/miniconda3:latest

# Set working directory
WORKDIR /workspace

# Copy environment.yaml
COPY environment.yaml ./

# Create the conda environment
RUN conda env create -f environment.yaml && conda clean -afy

# Activate environment by default
SHELL ["conda", "run", "-n", "exmed-bert", "/bin/bash", "-c"]

# Install pip packages (if not handled by conda)
# (Handled by environment.yaml)

# Set PATH for conda environment
ENV PATH /opt/conda/envs/exmed-bert/bin:$PATH

# Copy project files
COPY . /workspace

# Default command: start bash in the environment
CMD ["bash"]
