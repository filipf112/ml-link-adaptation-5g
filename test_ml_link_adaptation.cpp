#include <iostream>
#include <vector>
#include <iomanip>
#include "/home/fferenc/srsRAN_Project/lib/scheduler/support/ml_link_adaptation.h"
#include "/home/fferenc/srsRAN_Project/lib/scheduler/support/ml_link_adaptation.cpp" 

using namespace std;
using namespace srsran;

int main() {
    cout << "==================================================" << endl;
    cout << "   ML Link Adaptation C++ Standalone Simulator    " << endl;
    cout << "==================================================" << endl;

    OllaState state;
    
    // Simulate a user scenario: High speed, 2x2 MIMO, FR1, TDL-C channel
    double speed_kmph = 80.0;
    double num_antennas = 2.0;
    double carrier_band = 0.0;
    double channel_ordinal = 2.0;
    double bler_target_log10 = -1.0; // 10% target
    
    // Test: Changing SINR to see if it adapts correctly
    vector<double> sinr_trace = {5.0, 10.0, 15.0, 20.0, 25.0, 12.0, 3.0};
    
    cout << "Scenario: 80 km/h, 2x2 MIMO, FR1" << endl;
    cout << "Target BLER: 10%" << endl;
    cout << "--------------------------------------------------" << endl;
    cout << " SINR (dB) | EMA BLER | OLLA Offset | Selected MCS" << endl;
    cout << "--------------------------------------------------" << endl;
    
    for (double sinr : sinr_trace) {
        int mcs = select_mcs(sinr, speed_kmph, channel_ordinal, carrier_band, num_antennas, bler_target_log10, state);
        
        cout << " " << setw(9) << fixed << setprecision(1) << sinr 
             << " | " << setw(7) << fixed << setprecision(3) << state.bler_ema
             << "  | " << setw(11) << state.mcs_offset
             << " | " << setw(12) << mcs << endl;
             
        // Simulate sending 20 packets at this MCS
        for (int i=0; i<20; i++) {
            // Assume 1 in 10 fails (10% BLER)
            bool was_ack = (i % 10 != 0); 
            update_olla_state(state, was_ack);
        }
    }
    cout << "==================================================" << endl;
    return 0;
}
