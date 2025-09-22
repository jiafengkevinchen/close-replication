# Docker Instructions for Replication

This document provides step-by-step instructions for using Docker to set up the
environment for analysis

## 1. Install Docker

### Docker Installation

**Official installation guides:**
- **Ubuntu/Linux:** https://docs.docker.com/engine/install/ubuntu/
- **Windows:** https://docs.docker.com/desktop/install/windows-install/
- **macOS:** https://docs.docker.com/desktop/install/mac-install/

On Linux, use `sudo docker run hello-world` to check installation status

#### Linux post-install permissions

If you get `permission denied` running Docker without `sudo`, add your user to the
`docker` group and reload your session:

```bash
sudo usermod -aG docker $USER
newgrp docker
groups        # Verify that "docker" appears in the list
```

After this, you should be able to run commands such as `docker load` or
`docker compose` without prefixing them with `sudo`. Alternatively, keep using
`sudo docker ...` if you prefer not to modify group membership.

## 2. Set Up the Replication Environment

### Prerequisites
1. Obtain a MOSEK license file from https://www.mosek.com/license/request/
2. Save the license file as `mosek.lic` in the project directory

### Using Pre-built Docker Image (Fastest)

If you received a pre-built Docker image file (`eb-replication.tar`):

```bash
# Navigate to project directory
cd /path/to/close-replication-main

# Load the pre-built image
docker load < eb-replication.tar

# Verify the image was loaded
docker images

# Expected output:
# ❯ docker images
# REPOSITORY                   TAG       IMAGE ID       CREATED          SIZE
# replication-eb-replication   latest    3e160fc64e7f   18 minutes ago   1.02GB


# Place your MOSEK license file
# Copy your mosek.lic file to the project directory

# Start the container
docker compose up -d eb-replication

# Enter the container
docker compose exec eb-replication bash
```

### Building from Source (Alternative)

If you need to build the Docker image yourself:

```bash
# Navigate to project directory
cd /path/to/close-replication-main

# Place your MOSEK license file
# Copy your mosek.lic file to the project directory

# Build the Docker image (this takes 10-30 minutes)
docker compose build eb-replication

# Start the container
docker compose up -d eb-replication

# Enter the container
docker compose exec eb-replication bash
```

## 3. Working Inside the Container

Once inside the container, you can work as if you're in a normal Linux terminal and
proceed to README > Instructions to replicators

```bash

# Test the installation
# This also prints the max memory allowed in the container. Docker desktop might limit the max memory by default
# ensure that it's large enough (my computer is 24GB) and otherwise change settings in Docker Desktop > Settings > Resources.

root@26b7205c6672:/app$ python test.py
- Project '/app' loaded. [renv 1.1.4]
- The project is out-of-sync -- use `renv::status()` for details.
=== System Requirements ===
Total system memory: 23.4GB
Available memory: 22.4GB

=== Package Installation Check ===
Call: lprobust

Sample size (n)                              =    1000
Polynomial order for point estimation (p)    =    1
Order of derivative estimated (deriv)        =    0
Polynomial order for confidence interval (q) =    2
Kernel function                              =    Epanechnikov
Bandwidth method                             =    imse-dpi



Call:
	NULL

Data:  ( obs.);	Bandwidth 'bw' =

       x                 y
 Min.   :-3.0196   Min.   :0.000000
 1st Qu.:-1.2831   1st Qu.:0.000000
 Median : 0.4534   Median :0.000000
 Mean   : 0.4534   Mean   :0.003333
 3rd Qu.: 2.1898   3rd Qu.:0.000000
 Max.   : 3.9263   Max.   :0.998168


## Check a Monte Carlo draw

root@f7569e5a15bf:/app$ python check_monte_carlo.py --simulator_name npmle_by_bins --est_var kfr_top20_black_pooled_p25  --seed_number 381
- Project '/app' loaded. [renv 1.1.4]
- The project is out-of-sync -- use `renv::status()` for details.
Checking Monte Carlo outputs...
kfr_top20_black_pooled_p25: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:40<00:00, 40.62s/it]
--------------------------------------------------
Seed: 94682
Outcome variable: kfr_top20_black_pooled_p25
Simulator name: npmle_by_bins
Correlation between original and new Monte Carlo samples
(some differences may exist due to hardware):
--------------------------------------------------
                              Correlation     Intercept  Regression Coef
naive                            1.000000  7.914179e-08         1.000000
indep_npmle                      0.999999  5.910786e-07         0.999984
indep_gauss                      1.000000  1.264602e-06         0.999972
close_npmle                      1.000000  1.815419e-07         1.000001
close_gauss                      1.000000  3.764563e-07         0.999996
close_gauss_parametric           1.000000  6.241732e-07         0.999991
oracle                           1.000000 -1.941381e-07         1.000002
truth                            1.000000  2.749252e-08         1.000001
indep_npmle_nocov                1.000000  2.707387e-07         0.999998
indep_gauss_nocov                0.999999 -5.471731e-07         1.000028
close_npmle_nocov                1.000000  7.356839e-08         1.000002
close_gauss_nocov                1.000000  6.812757e-08         1.000001
close_gauss_parametric_nocov     1.000000  2.286719e-07         0.999998
true_covariate_fn                1.000000 -2.258983e-14         1.000000
truth_residualized               1.000000  1.367915e-08         1.000003
--------------------------------------------------
INFO:rpy2.rinterface_lib.embedded:Embedded R ended.
INFO:rpy2.rinterface_lib.embedded:Embedded R already ended.


## Check a Monte Carlo draw for coupled_bootstrap exercise

root@f7569e5a15bf:/app# python check_monte_carlo.py --simulator_name coupled_bootstrap-0.9 --est_var kfr_top20_black_pooled_p25  --seed_number 381
- Project '/app' loaded. [renv 1.1.4]
- The project is out-of-sync -- use `renv::status()` for details.
Checking Monte Carlo outputs...
kfr_top20_black_pooled_p25: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [01:05<00:00, 65.75s/it]
--------------------------------------------------
Seed: 94682
Outcome variable: kfr_top20_black_pooled_p25
Simulator name: coupled_bootstrap-0.9
Correlation between original and new Monte Carlo samples
(some differences may exist due to hardware):
--------------------------------------------------
                              Correlation     Intercept  Regression Coef
naive                                 1.0  8.857019e-18         1.000000
indep_npmle                           1.0 -1.170946e-09         1.000002
indep_gauss                           1.0 -7.904890e-16         1.000000
close_npmle                           1.0 -1.082923e-08         1.000000
close_gauss                           1.0 -2.014972e-15         1.000000
close_gauss_parametric                1.0 -1.798576e-09         1.000000
validation                            1.0  1.771404e-17         1.000000
validation_se                         1.0  0.000000e+00         1.000000
indep_npmle_nocov                     1.0  8.599778e-08         0.999997
indep_gauss_nocov                     1.0 -2.214255e-18         1.000000
close_npmle_nocov                     1.0  7.218302e-10         1.000000
close_gauss_nocov                     1.0 -1.067271e-15         1.000000
close_gauss_parametric_nocov          1.0  2.229859e-09         1.000000
--------------------------------------------------
INFO:rpy2.rinterface_lib.embedded:Embedded R ended.
INFO:rpy2.rinterface_lib.embedded:Embedded R already ended.

```

## 4. Troubleshooting and Rebuilding

### If Something Goes Wrong

**Container issues:**
```bash
# Stop and remove the container
docker compose down

# Remove the container completely
docker rm eb-replication-env

# Start fresh
docker compose up -d eb-replication
```

**Image corruption or need to rebuild:**
```bash
# Stop everything
docker compose down

# Remove the image (forces complete rebuild)
docker rmi $(docker compose images -q)

# Rebuild from scratch
docker compose build --no-cache eb-replication

# Start the new container
docker compose up -d eb-replication
```

**Memory issues:**
```bash
# Check Docker system resources
docker system info | grep -i memory

# Check container resource usage
docker stats eb-replication-env --no-stream

# If using Docker Desktop, increase memory allocation in Settings > Resources
```

### Common Issues

**"Permission denied" for shell scripts:**
```bash
# Inside the container, make scripts executable
chmod +x *.sh
```

**"Cannot find MOSEK license":**
- Ensure `mosek.lic` is in the project directory (same level as `docker-compose.yml`)
- Check that the file is not empty and is a valid MOSEK license

**Container hangs or won't start:**
- Check Docker logs: `docker compose logs eb-replication`
- For Docker Desktop: Restart Docker Desktop application
- For Docker CLI: Restart Docker daemon: `sudo systemctl restart docker`

**Build fails on ARM64 systems:**
- The Dockerfile uses `platform: linux/amd64` for MOSEK compatibility
- This works on ARM64 systems but runs slower due to emulation

## 5. Multiple Terminal Sessions

You can open multiple terminal sessions in the same container:

```bash
# Terminal 1 - Main work
docker compose exec eb-replication bash

# Terminal 2 - Monitoring (separate session, same container)
docker compose exec eb-replication bash
./monitor.sh

# Terminal 3 - Additional tasks
docker compose exec eb-replication bash
```

Each session is independent but shares the same filesystem and running processes.

## 6. File Access and Data Persistence

The container mounts several directories from your host system:
- `./results/` → `/app/results/` (analysis outputs)
- `./assets/` → `/app/assets/` (figures and tables)
- `./data/` → `/app/data/` (raw and processed data)
- `./mosek.lic` → `/app/mosek.lic` (MOSEK license - read-only)

**Important:** Files created inside these mounted directories will be available on your host system and persist even after the container is removed.

## 7. Container Lifecycle Management

```bash
# Check container status
docker compose ps

# Stop the container (keeps it for later use)
docker compose stop eb-replication

# Start a stopped container
docker compose start eb-replication

# Stop and remove the container (but keep the image)
docker compose down

# Remove the container and its volumes (careful - this removes data!)
docker compose down -v

# View container logs
docker compose logs eb-replication

# Follow logs in real-time
docker compose logs -f eb-replication
```

## 8. Resource Monitoring

```bash
# Monitor container resource usage (CPU, memory, I/O)
docker stats eb-replication-env

# One-time resource check
docker stats eb-replication-env --no-stream

# Check Docker system resource usage
docker system df

# View Docker system information
docker system info
```

## 9. Expected Workflow

Typical replication workflow:

```bash
# 1. Setup (one-time)
docker load < eb-replication.tar  # or docker compose build
docker compose up -d eb-replication

# 2. Enter container
docker compose exec eb-replication bash

# 3. Test installation
python test.py

# 4. Follow instructions in README.


# 5. When done, stop container
exit
docker compose down
```

The entire replication can be completed without ever leaving the container environment.
