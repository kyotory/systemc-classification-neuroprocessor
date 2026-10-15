#ifndef NEURO_POINT_TO_POINT_LINK_H
#define NEURO_POINT_TO_POINT_LINK_H

#include <systemc>
#include <cstddef>
#include <stdexcept>

SC_MODULE(PointToPointLink) {
    PointToPointLink(sc_core::sc_module_name name,
                     const sc_core::sc_time& transaction_latency)
        : sc_module(name), latency_(transaction_latency) {
        if (latency_ <= sc_core::SC_ZERO_TIME) {
            throw std::invalid_argument("Point-to-point link latency must be positive");
        }
    }

    void transaction() {
        ++transactions_;
        sc_core::wait(latency_);
    }

    void record_transactions(std::size_t count) { transactions_ += count; }

    std::size_t transactions() const { return transactions_; }
    const sc_core::sc_time& latency() const { return latency_; }

private:
    sc_core::sc_time latency_;
    std::size_t transactions_ = 0;
};

#endif
