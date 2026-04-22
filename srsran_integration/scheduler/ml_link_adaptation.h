#pragma once

#include <cstdint>
#include <cmath>
#include <algorithm>

namespace srsran {

struct OllaState {
    double bler_ema         = 0.10;
    int    consecutive_nack = 0;
    int    mcs_offset       = 0;
};

// Converts 3GPP CQI (1-15) to approximate SINR (dB)
double cqi_to_sinr_db(float cqi);

// Update OLLA EMA (Call when HARQ arrives)
void update_olla_state(OllaState& state, bool last_was_ack);

// Main ML-based MCS selection wrapper
int select_mcs(double measured_sinr_db,
               double measured_speed_kmph,
               double channel_ordinal, // V2 update
               double carrier_band,    // V2
               double num_antennas,    // V2
               double bler_target_log10,
               OllaState& state);

} // namespace srsran
