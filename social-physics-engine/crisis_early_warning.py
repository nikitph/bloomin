import numpy as np
from typing import Dict, List, Tuple
from gr_core import constitutional_metric

class SocialCrisisMonitor:
    """
    A real-time monitor for detecting event-horizon proximity 
    and gravitational wave patterns in social systems.
    """
    def __init__(self, rs_base: float = 0.217, eta_inf: float = 0.57):
        self.rs = rs_base
        self.eta_inf = eta_inf
        self.legitimacy_buffer = []
        self.max_buffer = 100
        
    def check_redshift(self, state: Dict, boundaries: List) -> Dict:
        """
        Measure the 'Social Redshift' z at the current state.
        z = (eta_inf - eta_eff) / eta_eff
        """
        theta = np.array([state.get('child_safety_risk', 1.0), state.get('time_until_deadline', 0.0)])
        g = constitutional_metric(theta, boundaries)
        # We use the radial component (risk) as the primary redshift driver
        eta_eff = 1.0 / np.sqrt(g[0,0])
        
        redshift = (self.eta_inf - eta_eff) / max(0.001, eta_eff)
        
        status = "SAFE"
        if redshift > 5.0: status = "CRITICAL"
        elif redshift > 1.0: status = "WARNING"
        
        return {
            'redshift': float(redshift),
            'eta_eff': float(eta_eff),
            'status': status,
            'distance_to_horizon': float(theta[0] - self.rs)
        }

    def detect_grav_waves(self, current_legitimacy: float) -> Dict:
        """
        Detect quadrupole oscillations in the legitimacy field.
        """
        self.legitimacy_buffer.append(current_legitimacy)
        if len(self.legitimacy_buffer) > self.max_buffer:
            self.legitimacy_buffer.pop(0)
            
        if len(self.legitimacy_buffer) < 20:
            return {'gw_intensity': 0.0, 'peak_freq': 0.0, 'status': "CALIBRATING"}
            
        # FFT to find peaks
        signal = np.array(self.legitimacy_buffer) - np.mean(self.legitimacy_buffer)
        fft_vals = np.abs(np.fft.rfft(signal))
        freqs = np.fft.rfftfreq(len(signal))
        
        peak_idx = np.argmax(fft_vals)
        peak_freq = freqs[peak_idx]
        intensity = fft_vals[peak_idx]
        
        gw_status = "STABLE"
        if intensity > 1.0 and peak_freq > 0.1:
            gw_status = "DETECTED"
            if peak_freq > 0.4: gw_status = "CHIRP_PHASE" # Fast oscillations = near merger
            
        return {
            'gw_intensity': float(intensity),
            'peak_freq': float(peak_freq),
            'status': gw_status
        }

if __name__ == "__main__":
    print("Testing Social Crisis Early Warning System...")
    monitor = SocialCrisisMonitor()
    
    # 1. Test Redshift
    safe_state = {'child_safety_risk': 0.1, 'time_until_deadline': 10.0}
    # Boundaries from engine
    from engine import SocialPhysicsEngine
    engine = SocialPhysicsEngine(num_agents=1)
    boundaries = [engine.role_boundaries['parent']]
    
    print(f"Safe State: {monitor.check_redshift(safe_state, boundaries)}")
    
    danger_state = {'child_safety_risk': 0.7, 'time_until_deadline': 2.0}
    print(f"Danger State: {monitor.check_redshift(danger_state, boundaries)}")
    
    # 2. Test GW Detection
    print("\nSimulating GW Chirp...")
    for i in range(50):
        # Increasing frequency chirp, higher amplitude
        f = 0.1 + 0.01 * i
        legitimacy = 0.5 * np.sin(2 * np.pi * f * i)
        gw = monitor.detect_grav_waves(legitimacy)
        if i % 10 == 0:
            print(f"  t={i}: GW {gw['status']} (f={gw['peak_freq']:.3f}, intensity={gw['gw_intensity']:.3f})")
