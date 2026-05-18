#!/bin/bash
# =================================================================
#  srsRAN ML Link Adaptation — Live Test Suite
# =================================================================
#
#  Prerequisites:
#    1. Open5GS 5GC running (docker)
#    2. gNB running with ML scheduler
#    3. UE attached and PDU session established
#    4. Network namespace 'ue1' configured
#    5. Routing configured:
#       sudo ip ro add 10.45.0.0/16 via 10.153.1.2
#       sudo ip netns exec ue1 ip route add default via 10.45.1.1 dev tun_srsue
#
#  Usage:
#    sudo bash run_srsran_tests.sh
#
# =================================================================

set -e

UE_NS="ue1"
SERVER_IP="10.45.1.1"
RESULTS_DIR="results/srsran"
LOG_FILE="$RESULTS_DIR/test_results.txt"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

mkdir -p "$RESULTS_DIR"

echo "=================================================================" | tee "$LOG_FILE"
echo "  srsRAN ML Link Adaptation — Live Test Suite" | tee -a "$LOG_FILE"
echo "  $(date)" | tee -a "$LOG_FILE"
echo "=================================================================" | tee -a "$LOG_FILE"

# ── Pre-flight check ───────────────────────────────────────────────
echo -e "\n${BLUE}[0] Pre-flight check...${NC}" | tee -a "$LOG_FILE"

if ! ip netns exec "$UE_NS" ping -c 1 -W 2 "$SERVER_IP" > /dev/null 2>&1; then
    echo "ERROR: Cannot reach $SERVER_IP from $UE_NS namespace!"
    echo "Make sure UE is attached and routing is configured."
    exit 1
fi
echo "   ✓ Connectivity OK" | tee -a "$LOG_FILE"

# Check iperf3 server
if ! ip netns exec "$UE_NS" which iperf3 > /dev/null 2>&1; then
    IPERF3=$(which iperf3)
else
    IPERF3="iperf3"
fi
echo "   ✓ iperf3: $IPERF3" | tee -a "$LOG_FILE"

# ── Test 1: Baseline Ping ──────────────────────────────────────────
echo -e "\n${GREEN}[1] Baseline Ping Test (500 packets)${NC}" | tee -a "$LOG_FILE"
echo "-----------------------------------------------------------------" | tee -a "$LOG_FILE"

ip netns exec "$UE_NS" ping -c 500 -i 0.01 "$SERVER_IP" 2>&1 | tail -3 | tee -a "$LOG_FILE"

# ── Test 2: UDP Throughput Sweep ───────────────────────────────────
echo -e "\n${GREEN}[2] UDP Throughput Sweep${NC}" | tee -a "$LOG_FILE"
echo "-----------------------------------------------------------------" | tee -a "$LOG_FILE"

# Start iperf3 server in background (in default namespace)
pkill -f "iperf3 -s" 2>/dev/null || true
sleep 1
iperf3 -s -D --logfile "$RESULTS_DIR/iperf3_server.log"
sleep 1

for BITRATE in 1M 5M 10M 20M; do
    echo -e "\n   ${YELLOW}--- UDP $BITRATE, 30 seconds ---${NC}" | tee -a "$LOG_FILE"
    ip netns exec "$UE_NS" $IPERF3 -c "$SERVER_IP" -u -b "$BITRATE" -t 30 \
        --json > "$RESULTS_DIR/udp_${BITRATE}.json" 2>&1

    # Extract summary
    python3 -c "
import json, sys
try:
    with open('$RESULTS_DIR/udp_${BITRATE}.json') as f:
        d = json.load(f)
    s = d.get('end', {}).get('sum', {})
    print(f'   Bitrate: {s.get(\"bits_per_second\", 0)/1e6:.2f} Mbps')
    print(f'   Jitter:  {s.get(\"jitter_ms\", 0):.3f} ms')
    print(f'   Loss:    {s.get(\"lost_percent\", 0):.1f}%')
    print(f'   Packets: {s.get(\"packets\", 0)}')
except Exception as e:
    print(f'   Error parsing JSON: {e}')
" 2>&1 | tee -a "$LOG_FILE"

    sleep 3
done

# ── Test 3: TCP Throughput ─────────────────────────────────────────
echo -e "\n${GREEN}[3] TCP Throughput Test (30 seconds)${NC}" | tee -a "$LOG_FILE"
echo "-----------------------------------------------------------------" | tee -a "$LOG_FILE"

ip netns exec "$UE_NS" $IPERF3 -c "$SERVER_IP" -t 30 \
    --json > "$RESULTS_DIR/tcp_30s.json" 2>&1

python3 -c "
import json
try:
    with open('$RESULTS_DIR/tcp_30s.json') as f:
        d = json.load(f)
    s = d.get('end', {}).get('sum_sent', {})
    r = d.get('end', {}).get('sum_received', {})
    print(f'   Sent:     {s.get(\"bits_per_second\", 0)/1e6:.2f} Mbps')
    print(f'   Received: {r.get(\"bits_per_second\", 0)/1e6:.2f} Mbps')
    print(f'   Retrans:  {s.get(\"retransmits\", \"N/A\")}')
except Exception as e:
    print(f'   Error: {e}')
" 2>&1 | tee -a "$LOG_FILE"

sleep 3

# ── Test 4: Ping Under Load ───────────────────────────────────────
echo -e "\n${GREEN}[4] Latency Under Load (ping + iperf3 10M)${NC}" | tee -a "$LOG_FILE"
echo "-----------------------------------------------------------------" | tee -a "$LOG_FILE"

# Start background iperf3 UDP traffic
ip netns exec "$UE_NS" $IPERF3 -c "$SERVER_IP" -u -b 10M -t 20 \
    > "$RESULTS_DIR/background_traffic.log" 2>&1 &
IPERF_PID=$!
sleep 2

# Run ping during load
echo "   Ping during 10M UDP load:" | tee -a "$LOG_FILE"
ip netns exec "$UE_NS" ping -c 100 -i 0.1 "$SERVER_IP" 2>&1 | tail -3 | tee -a "$LOG_FILE"

wait $IPERF_PID 2>/dev/null || true
sleep 2

# ── Test 5: Ping Without Load (reference) ─────────────────────────
echo -e "\n   Ping without load (reference):" | tee -a "$LOG_FILE"
ip netns exec "$UE_NS" ping -c 100 -i 0.1 "$SERVER_IP" 2>&1 | tail -3 | tee -a "$LOG_FILE"

# ── Cleanup ────────────────────────────────────────────────────────
pkill -f "iperf3 -s" 2>/dev/null || true

echo -e "\n${GREEN}=================================================================${NC}" | tee -a "$LOG_FILE"
echo "  All tests complete!" | tee -a "$LOG_FILE"
echo "  Results saved to: $RESULTS_DIR/" | tee -a "$LOG_FILE"
echo "  " | tee -a "$LOG_FILE"
echo "  Next steps:" | tee -a "$LOG_FILE"
echo "    1. Copy UE log: cp /tmp/ue.log $RESULTS_DIR/" | tee -a "$LOG_FILE"
echo "    2. Parse logs:  python3 parse_srsran_logs.py $RESULTS_DIR/ue.log" | tee -a "$LOG_FILE"
echo "=================================================================" | tee -a "$LOG_FILE"
