#include "ml_link_adaptation.h"
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace srsran {

double cqi_to_sinr_db(float cqi) {
    if (cqi <= 1.0f) return -6.7;
    if (cqi <= 2.0f) return -4.0;
    if (cqi <= 3.0f) return -2.0;
    if (cqi <= 4.0f) return 0.0;
    if (cqi <= 5.0f) return 2.0;
    if (cqi <= 6.0f) return 4.0;
    if (cqi <= 7.0f) return 6.0;
    if (cqi <= 8.0f) return 8.0;
    if (cqi <= 9.0f) return 10.0;
    if (cqi <= 10.0f) return 12.0;
    if (cqi <= 11.0f) return 14.0;
    if (cqi <= 12.0f) return 16.0;
    if (cqi <= 13.0f) return 18.0;
    if (cqi <= 14.0f) return 20.0;
    return 22.0;
}


static constexpr int VALID_MCS[9] = {3, 4, 9, 11, 14, 17, 20, 24, 25};
static constexpr int N_MCS = 9;


static int nearest_valid_mcs(int raw) {
    int best = VALID_MCS[0];
    for (int i = 1; i < N_MCS; ++i) {
        if (std::abs(VALID_MCS[i] - raw) < std::abs(best - raw))
            best = VALID_MCS[i];
    }
    return best;
}

static int select_mcs_base(double measured_sinr_db,
                           double measured_speed_kmph,
                           double channel_ordinal,
                           double carrier_band,
                           double num_antennas,
                           double bler_target_log10) {
    if (measured_sinr_db <= 4.30839133) {
        if (measured_sinr_db <= -0.96642464) {
            if (measured_sinr_db <= -2.48702228) {
                return 3;
            } else {
                if (channel_ordinal <= 0.50000000) {
                    if (measured_speed_kmph <= 70.19161415) {
                        if (num_antennas <= 1.50000000) {
                            return 3;
                        } else {
                            if (measured_speed_kmph <= 18.02309258) {
                                return 4;
                            } else {
                                return 3;
                            }
                        }
                    } else {
                        return 3;
                    }
                } else {
                    return 3;
                }
            }
        } else {
            if (channel_ordinal <= 0.50000000) {
                if (measured_sinr_db <= 1.26274550) {
                    if (carrier_band <= 0.50000000) {
                        if (measured_speed_kmph <= 68.53525734) {
                            return 4;
                        } else {
                            return 3;
                        }
                    } else {
                        if (measured_speed_kmph <= 12.77672052) {
                            return 3;
                        } else {
                            return 3;
                        }
                    }
                } else {
                    return 4;
                }
            } else {
                if (measured_speed_kmph <= 0.55807936) {
                    if (measured_sinr_db <= 1.56168413) {
                        if (measured_sinr_db <= 0.41184593) {
                            return 4;
                        } else {
                            return 3;
                        }
                    } else {
                        return 4;
                    }
                } else {
                    if (measured_speed_kmph <= 6.58179379) {
                        if (measured_speed_kmph <= 3.62142491) {
                            return 3;
                        } else {
                            return 3;
                        }
                    } else {
                        if (measured_speed_kmph <= 21.72451115) {
                            if (measured_speed_kmph <= 18.92787075) {
                                return 3;
                            } else {
                                return 3;
                            }
                        } else {
                            return 3;
                        }
                    }
                }
            }
        }
    } else {
        if (measured_sinr_db <= 12.56060553) {
            if (measured_sinr_db <= 6.17890668) {
                if (measured_speed_kmph <= 3.99698102) {
                    if (channel_ordinal <= 0.50000000) {
                        if (measured_speed_kmph <= 0.41157654) {
                            return 9;
                        } else {
                            return 4;
                        }
                    } else {
                        return 4;
                    }
                } else {
                    if (measured_sinr_db <= 4.54392958) {
                        if (channel_ordinal <= 2.00000000) {
                            return 9;
                        } else {
                            return 4;
                        }
                    } else {
                        if (measured_speed_kmph <= 124.33786774) {
                            return 4;
                        } else {
                            if (channel_ordinal <= 2.00000000) {
                                return 4;
                            } else {
                                return 3;
                            }
                        }
                    }
                }
            } else {
                if (measured_sinr_db <= 8.82764578) {
                    if (channel_ordinal <= 0.50000000) {
                        if (measured_sinr_db <= 7.05531454) {
                            if (measured_speed_kmph <= 1.00115356) {
                                if (num_antennas <= 1.50000000) {
                                    return 11;
                                } else {
                                    return 9;
                                }
                            } else {
                                if (measured_sinr_db <= 6.62034631) {
                                    return 9;
                                } else {
                                    if (measured_sinr_db <= 6.76861310) {
                                        return 11;
                                    } else {
                                        return 9;
                                    }
                                }
                            }
                        } else {
                            if (measured_speed_kmph <= 12.73514843) {
                                if (num_antennas <= 1.50000000) {
                                    return 11;
                                } else {
                                    if (measured_speed_kmph <= 4.53145599) {
                                        return 14;
                                    } else {
                                        return 11;
                                    }
                                }
                            } else {
                                if (measured_speed_kmph <= 118.55266190) {
                                    return 11;
                                } else {
                                    return 9;
                                }
                            }
                        }
                    } else {
                        if (num_antennas <= 1.50000000) {
                            if (measured_speed_kmph <= 1.50999677) {
                                if (measured_sinr_db <= 7.24577546) {
                                    return 9;
                                } else {
                                    return 11;
                                }
                            } else {
                                if (measured_sinr_db <= 7.73050022) {
                                    return 9;
                                } else {
                                    if (measured_speed_kmph <= 34.94818306) {
                                        return 9;
                                    } else {
                                        return 9;
                                    }
                                }
                            }
                        } else {
                            if (channel_ordinal <= 2.50000000) {
                                if (measured_sinr_db <= 7.53271031) {
                                    if (carrier_band <= 0.50000000) {
                                        return 9;
                                    } else {
                                        return 4;
                                    }
                                } else {
                                    if (measured_speed_kmph <= 79.28376770) {
                                        return 9;
                                    } else {
                                        return 4;
                                    }
                                }
                            } else {
                                if (measured_speed_kmph <= 128.31045914) {
                                    return 4;
                                } else {
                                    return 3;
                                }
                            }
                        }
                    }
                } else {
                    if (measured_sinr_db <= 10.87900114) {
                        if (num_antennas <= 1.50000000) {
                            if (measured_speed_kmph <= 15.18932247) {
                                if (measured_sinr_db <= 10.03593397) {
                                    return 11;
                                } else {
                                    return 11;
                                }
                            } else {
                                return 11;
                            }
                        } else {
                            if (channel_ordinal <= 0.50000000) {
                                if (carrier_band <= 0.50000000) {
                                    if (measured_sinr_db <= 9.06382465) {
                                        return 11;
                                    } else {
                                        return 14;
                                    }
                                } else {
                                    if (measured_sinr_db <= 9.91763783) {
                                        return 11;
                                    } else {
                                        return 11;
                                    }
                                }
                            } else {
                                if (measured_speed_kmph <= 36.81468201) {
                                    return 11;
                                } else {
                                    return 9;
                                }
                            }
                        }
                    } else {
                        if (measured_speed_kmph <= 109.10931778) {
                            if (channel_ordinal <= 0.50000000) {
                                if (measured_speed_kmph <= 4.72279215) {
                                    return 17;
                                } else {
                                    return 14;
                                }
                            } else {
                                if (num_antennas <= 1.50000000) {
                                    return 14;
                                } else {
                                    if (measured_speed_kmph <= 32.20698929) {
                                        return 11;
                                    } else {
                                        return 9;
                                    }
                                }
                            }
                        } else {
                            if (measured_sinr_db <= 11.45182133) {
                                if (measured_sinr_db <= 11.13978100) {
                                    return 11;
                                } else {
                                    return 9;
                                }
                            } else {
                                if (measured_speed_kmph <= 115.86206055) {
                                    return 11;
                                } else {
                                    return 11;
                                }
                            }
                        }
                    }
                }
            }
        } else {
            if (measured_sinr_db <= 17.16288471) {
                if (channel_ordinal <= 0.50000000) {
                    if (measured_speed_kmph <= 112.94168854) {
                        if (measured_sinr_db <= 16.70649338) {
                            if (measured_speed_kmph <= 109.61881256) {
                                return 17;
                            } else {
                                return 11;
                            }
                        } else {
                            if (measured_sinr_db <= 16.77868176) {
                                return 20;
                            } else {
                                return 17;
                            }
                        }
                    } else {
                        if (carrier_band <= 0.50000000) {
                            return 17;
                        } else {
                            if (measured_sinr_db <= 15.33669233) {
                                return 11;
                            } else {
                                return 14;
                            }
                        }
                    }
                } else {
                    if (measured_speed_kmph <= 33.18185043) {
                        if (num_antennas <= 1.50000000) {
                            if (measured_sinr_db <= 14.21487474) {
                                return 14;
                            } else {
                                if (measured_speed_kmph <= 2.00723720) {
                                    if (measured_sinr_db <= 15.62707567) {
                                        return 17;
                                    } else {
                                        return 17;
                                    }
                                } else {
                                    return 17;
                                }
                            }
                        } else {
                            if (measured_sinr_db <= 14.94774055) {
                                if (measured_speed_kmph <= 8.50468922) {
                                    return 14;
                                } else {
                                    return 11;
                                }
                            } else {
                                if (measured_sinr_db <= 16.51481247) {
                                    return 14;
                                } else {
                                    return 14;
                                }
                            }
                        }
                    } else {
                        if (num_antennas <= 1.50000000) {
                            if (measured_sinr_db <= 14.59946728) {
                                if (measured_speed_kmph <= 117.04510117) {
                                    return 14;
                                } else {
                                    return 11;
                                }
                            } else {
                                return 17;
                            }
                        } else {
                            if (measured_sinr_db <= 14.08268023) {
                                if (measured_sinr_db <= 13.07105064) {
                                    return 11;
                                } else {
                                    return 9;
                                }
                            } else {
                                return 11;
                            }
                        }
                    }
                }
            } else {
                if (measured_speed_kmph <= 58.35043716) {
                    if (measured_sinr_db <= 23.22297764) {
                        if (measured_sinr_db <= 18.82577515) {
                            if (channel_ordinal <= 0.50000000) {
                                return 20;
                            } else {
                                if (carrier_band <= 0.50000000) {
                                    if (num_antennas <= 1.50000000) {
                                        return 17;
                                    } else {
                                        return 17;
                                    }
                                } else {
                                    if (measured_speed_kmph <= 11.29361582) {
                                        return 17;
                                    } else {
                                        if (num_antennas <= 1.50000000) {
                                            return 17;
                                        } else {
                                            return 14;
                                        }
                                    }
                                }
                            }
                        } else {
                            if (measured_speed_kmph <= 8.28355694) {
                                if (measured_sinr_db <= 20.62684345) {
                                    if (channel_ordinal <= 0.50000000) {
                                        return 24;
                                    } else {
                                        return 20;
                                    }
                                } else {
                                    if (channel_ordinal <= 0.50000000) {
                                        return 25;
                                    } else {
                                        if (num_antennas <= 1.50000000) {
                                            if (carrier_band <= 0.50000000) {
                                                if (measured_speed_kmph <= 2.07343048) {
                                                    return 24;
                                                } else {
                                                    return 24;
                                                }
                                            } else {
                                                return 25;
                                            }
                                        } else {
                                            return 20;
                                        }
                                    }
                                }
                            } else {
                                if (channel_ordinal <= 0.50000000) {
                                    if (measured_speed_kmph <= 34.18718910) {
                                        if (measured_speed_kmph <= 14.19350100) {
                                            return 24;
                                        } else {
                                            if (measured_sinr_db <= 22.10802555) {
                                                return 20;
                                            } else {
                                                return 20;
                                            }
                                        }
                                    } else {
                                        return 24;
                                    }
                                } else {
                                    if (num_antennas <= 1.50000000) {
                                        if (measured_speed_kmph <= 15.33051395) {
                                            return 20;
                                        } else {
                                            return 20;
                                        }
                                    } else {
                                        if (channel_ordinal <= 2.50000000) {
                                            if (carrier_band <= 0.50000000) {
                                                return 20;
                                            } else {
                                                return 14;
                                            }
                                        } else {
                                            return 14;
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        if (measured_speed_kmph <= 12.15252781) {
                            if (channel_ordinal <= 0.50000000) {
                                return 25;
                            } else {
                                if (channel_ordinal <= 2.50000000) {
                                    if (num_antennas <= 1.50000000) {
                                        return 25;
                                    } else {
                                        if (measured_sinr_db <= 28.11477184) {
                                            if (carrier_band <= 0.50000000) {
                                                return 20;
                                            } else {
                                                return 20;
                                            }
                                        } else {
                                            return 24;
                                        }
                                    }
                                } else {
                                    if (carrier_band <= 0.50000000) {
                                        return 24;
                                    } else {
                                        if (measured_speed_kmph <= 0.55706936) {
                                            return 25;
                                        } else {
                                            return 24;
                                        }
                                    }
                                }
                            }
                        } else {
                            if (carrier_band <= 0.50000000) {
                                if (measured_sinr_db <= 25.37801456) {
                                    if (measured_sinr_db <= 24.49721813) {
                                        if (measured_sinr_db <= 23.74157238) {
                                            return 24;
                                        } else {
                                            if (measured_sinr_db <= 24.33358669) {
                                                return 20;
                                            } else {
                                                return 24;
                                            }
                                        }
                                    } else {
                                        if (measured_sinr_db <= 25.08034420) {
                                            return 20;
                                        } else {
                                            return 20;
                                        }
                                    }
                                } else {
                                    return 24;
                                }
                            } else {
                                if (measured_speed_kmph <= 36.00370216) {
                                    if (channel_ordinal <= 0.50000000) {
                                        if (measured_sinr_db <= 26.15872860) {
                                            return 24;
                                        } else {
                                            return 20;
                                        }
                                    } else {
                                        if (measured_speed_kmph <= 28.45121098) {
                                            if (measured_sinr_db <= 25.85913944) {
                                                return 20;
                                            } else {
                                                return 20;
                                            }
                                        } else {
                                            return 17;
                                        }
                                    }
                                } else {
                                    if (channel_ordinal <= 2.50000000) {
                                        return 17;
                                    } else {
                                        return 14;
                                    }
                                }
                            }
                        }
                    }
                } else {
                    if (channel_ordinal <= 0.50000000) {
                        if (measured_speed_kmph <= 121.84026337) {
                            if (measured_sinr_db <= 24.34941578) {
                                return 20;
                            } else {
                                if (num_antennas <= 1.50000000) {
                                    return 24;
                                } else {
                                    return 20;
                                }
                            }
                        } else {
                            if (carrier_band <= 0.50000000) {
                                if (measured_sinr_db <= 21.12915993) {
                                    return 17;
                                } else {
                                    if (measured_sinr_db <= 24.46254063) {
                                        return 20;
                                    } else {
                                        return 20;
                                    }
                                }
                            } else {
                                if (measured_speed_kmph <= 133.05709076) {
                                    return 17;
                                } else {
                                    return 14;
                                }
                            }
                        }
                    } else {
                        if (carrier_band <= 0.50000000) {
                            if (num_antennas <= 1.50000000) {
                                if (measured_speed_kmph <= 125.73830414) {
                                    return 17;
                                } else {
                                    return 14;
                                }
                            } else {
                                if (channel_ordinal <= 2.50000000) {
                                    if (measured_sinr_db <= 25.79088306) {
                                        return 14;
                                    } else {
                                        return 14;
                                    }
                                } else {
                                    return 11;
                                }
                            }
                        } else {
                            if (measured_speed_kmph <= 102.03525162) {
                                return 11;
                            } else {
                                return 11;
                            }
                        }
                    }
                }
            }
        }
    }
}

void update_olla_state(OllaState& state, bool last_was_ack) {
    constexpr double kAlpha = 0.02;
    state.bler_ema = kAlpha * (last_was_ack ? 0.0 : 1.0)
                   + (1.0 - kAlpha) * state.bler_ema;

    state.consecutive_nack = last_was_ack ? 0
                           : state.consecutive_nack + 1;
}

int select_mcs(double measured_sinr_db,
               double measured_speed_kmph,
               double channel_ordinal,
               double carrier_band,
               double num_antennas,
               double bler_target_log10,
               OllaState& state) {

    if (state.consecutive_nack > 100) {
        return VALID_MCS[0];
    }

    double bler_target = std::pow(10.0, bler_target_log10);
    if (state.bler_ema > bler_target) {
        state.mcs_offset = std::max(state.mcs_offset - 1, -3);
    } else if (state.bler_ema < bler_target * 0.5) {
        state.mcs_offset = std::min(state.mcs_offset + 1, 2);
    }

    int base = select_mcs_base(measured_sinr_db, measured_speed_kmph,
                               channel_ordinal, carrier_band,
                               num_antennas, bler_target_log10);
    int adjusted = base + state.mcs_offset;
    adjusted = std::clamp(adjusted, VALID_MCS[0], VALID_MCS[N_MCS-1]);
    return nearest_valid_mcs(adjusted);
}

} // namespace srsran
