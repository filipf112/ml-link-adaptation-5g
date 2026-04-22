# srsRAN 5G NR Integration — ML Link Adaptation

This directory contains the source code modifications required to deploy the trained ML-based Link Adaptation policy into a live 5G NR Standalone (SA) network using [srsRAN Project](https://github.com/srsran/srsRAN_Project) (gNB) and [srsRAN 4G](https://github.com/srsran/srsRAN_4G) (srsUE).

## Architecture

```
┌──────────┐   ZMQ IQ Samples   ┌──────────────────┐  GTP-U / SCTP  ┌─────────────┐
│  srsUE   │◄──────────────────►│   srsRAN gNB      │◄──────────────►│  Open5GS 5GC│
│(srsRAN4G)│  tcp:2000 / 2001   │  (ML Scheduler)   │  10.153.1.x    │  (Docker)   │
└──────────┘                    └──────────────────┘                 └─────────────┘
                                         │
                                  ML Decision Tree
                                  (MCS Selection)
```

## File Placement

The files must be placed in the srsRAN Project source tree as follows:

### New Files (ML Policy)
| Source | Destination in srsRAN Project |
|--------|-------------------------------|
| `scheduler/ml_link_adaptation.cpp` | `lib/scheduler/support/ml_link_adaptation.cpp` |
| `scheduler/ml_link_adaptation.h` | `lib/scheduler/support/ml_link_adaptation.h` |

### Modified Files (Integration)
| Source | Destination in srsRAN Project |
|--------|-------------------------------|
| `scheduler/ue_link_adaptation_controller.cpp` | `lib/scheduler/ue_context/ue_link_adaptation_controller.cpp` |
| `scheduler/ue_link_adaptation_controller.h` | `lib/scheduler/ue_context/ue_link_adaptation_controller.h` |
| `scheduler/CMakeLists.txt` | `lib/scheduler/support/CMakeLists.txt` |

### Configuration Files
| Source | Destination |
|--------|-------------|
| `configs/gnb_ml_zmq.yaml` | srsRAN Project `configs/` directory |
| `configs/ue_ml_zmq.conf` | srsRAN 4G root directory |

### Docker (Open5GS 5G Core)
| Source | Destination |
|--------|-------------|
| `docker/docker-compose.yml` | srsRAN Project `docker/` directory |
| `docker/open5gs.env` | srsRAN Project `docker/open5gs/` directory |

## Build Instructions

### Prerequisites
- srsRAN Project (commit `4bf1543936` or compatible)
- srsRAN 4G (commit `6bcbd9e5b` or compatible)
- Docker & Docker Compose (for Open5GS 5G Core)
- ZeroMQ development libraries (`libzmq3-dev`)

### 1. Copy ML Policy Files
```bash
cp scheduler/ml_link_adaptation.cpp  <srsRAN_Project>/lib/scheduler/support/
cp scheduler/ml_link_adaptation.h    <srsRAN_Project>/lib/scheduler/support/
cp scheduler/ue_link_adaptation_controller.cpp <srsRAN_Project>/lib/scheduler/ue_context/
cp scheduler/ue_link_adaptation_controller.h   <srsRAN_Project>/lib/scheduler/ue_context/
cp scheduler/CMakeLists.txt          <srsRAN_Project>/lib/scheduler/support/
```

### 2. Rebuild gNB
```bash
cd <srsRAN_Project>/build
cmake ..
make -j$(nproc)
```

### 3. Start the 5G Network

```bash
# Terminal 1: Create UE network namespace
sudo ip netns add ue1

# Terminal 1: Start 5G Core
sudo docker compose -f <srsRAN_Project>/docker/docker-compose.yml up -d 5gc

# Terminal 2: Start gNB
sudo <srsRAN_Project>/build/apps/gnb/gnb -c configs/gnb_ml_zmq.yaml

# Terminal 3: Start UE
sudo <srsRAN_4G>/build/srsue/src/srsue configs/ue_ml_zmq.conf
```

### 4. Validate

After `PDU Session Establishment successful` appears:
```bash
# Setup routing
sudo ip ro add 10.45.0.0/16 via 10.153.1.2
sudo ip netns exec ue1 ip route add default via 10.45.1.1 dev tun_srsue

# Ping test
sudo ip netns exec ue1 ping 10.45.1.1
```

## How It Works

The ML policy replaces the standard CQI-to-MCS mapping in the DL scheduler path:

1. **Input**: The UE reports wideband CQI via PUCCH → mapped to estimated SINR via `cqi_to_sinr_db()`
2. **Decision**: `select_mcs()` traverses a depth-10 decision tree trained on 3GPP TDL channel simulations
3. **Safety**: An OLLA state tracker (`OllaState`) adjusts MCS ±3 based on observed BLER
4. **Output**: Selected MCS is clamped to the configured `qam64` table range

The decision tree executes in **O(1)** with deterministic branching — no heap allocations, no floating-point exceptions — suitable for real-time L2 scheduling within the 1ms slot budget.
