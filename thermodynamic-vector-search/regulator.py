from operators import WaveOperator, HeatOperator
from math import log2

class ThermodynamicRegulator:
    """
    Control logic that decides when to switch operators
    based on spectral monitoring and energy budgets
    """
    
    def __init__(self, 
                 epsilon_critical=0.01,
                 energy_budget=1000,
                 resonance_threshold=0.1,
                 convergence_threshold=1e-4):
        
        self.epsilon_critical = epsilon_critical
        self.energy_budget = energy_budget
        self.resonance_threshold = resonance_threshold
        self.convergence_threshold = convergence_threshold
        
        # State tracking
        self.energy_spent = 0
        self.current_phase = None
        self.phase_start_time = 0
    
    def should_transition(self, field_state, operator, monitor):
        """
        Decide if we should switch operators
        
        Returns: (should_switch, reason, next_operator_type)
        """
        current_gap = monitor.compute_spectral_gap(field_state)
        
        # Check energy budget
        if self.energy_spent >= self.energy_budget:
            return True, "energy_exhausted", None
        
        # Phase-specific logic
        if isinstance(operator, WaveOperator):
            return self._check_wave_transition(
                field_state, operator, monitor, current_gap
            )
        
        elif isinstance(operator, HeatOperator):
            return self._check_heat_transition(
                field_state, operator, monitor, current_gap
            )
        
        return False, None, None
    
    def _check_wave_transition(self, field_state, operator, 
                               monitor, current_gap):
        """
        Transition logic for Wave phase
        """
        # Check 1: Spectral gap collapse
        if current_gap < self.epsilon_critical:
            return True, "gap_collapse", "heat"
        
        # Check 2: Strong resonance detected
        has_resonance, location, strength = operator.detect_resonance(
            field_state, threshold=self.resonance_threshold
        )
        
        if has_resonance:
            # Resonance found -> zoom in with heat diffusion
            return True, "resonance_detected", "heat"
        
        # Check 3: Timeout (wave not finding anything)
        time_in_phase = field_state.time - self.phase_start_time
        if time_in_phase > 100:  # Max wave time
            return True, "wave_timeout", "heat"
        
        # Continue wave scan
        return False, None, None
    
    def _check_heat_transition(self, field_state, operator, 
                               monitor, current_gap):
        """
        Transition logic for Heat phase
        """
        # Check 1: Convergence
        if operator.has_converged(field_state, 
                                  self.convergence_threshold):
            return True, "converged", "done"
        
        # Check 2: Spectral gap collapse (shouldn't happen in local region)
        # Note: If gap collapses in refinement, it's usually bad, but maybe we just finish
        if current_gap < self.epsilon_critical:
            # Something wrong, or processed finished, emergency exit
            return True, "gap_collapse_in_heat", "done"
        
        # Check 3: Timeout
        time_in_phase = field_state.time - self.phase_start_time
        if time_in_phase > 100:  # Max heat time
            return True, "heat_timeout", "done"
        
        # Continue heat diffusion
        return False, None, None
    
    def estimate_work(self, operator, field_state):
        """
        Estimate computational work for next steps
        
        Returns: estimated iteration count
        """
        if isinstance(operator, WaveOperator):
            # Wave: estimate based on wave speed and grid size
            grid_size = field_state.phi.size
            if grid_size > 0:
                 wave_hops = log2(grid_size)  # Exponential spread
            else:
                 wave_hops = 1
            return int(wave_hops * 10)  # ~10 steps per hop
        
        elif isinstance(operator, HeatOperator):
            # Heat: estimate based on spectral gap
            gap = field_state.spectral_gap
            if gap > 0:
                mixing_time = 1.0 / gap
                return int(mixing_time * 2)  # Safety factor
            else:
                return 1000  # Fallback
        
        return 100  # Default
    
    def select_operator(self, next_type, field_state):
        """
        Create appropriate operator for next phase
        """
        if next_type == "wave":
            return WaveOperator(
                wave_speed=1.0,
                damping=0.5,
                dt=0.01
            )
        
        elif next_type == "heat":
            return HeatOperator(
                diffusion_coeff=0.1,
                dt=0.01,
                anisotropic=True
            )
        
        return None
    
    def start_phase(self, phase_name, field_state):
        """Track phase transitions"""
        self.current_phase = phase_name
        self.phase_start_time = field_state.time
    
    def record_energy(self, spent):
        """Track energy expenditure"""
        self.energy_spent += spent
