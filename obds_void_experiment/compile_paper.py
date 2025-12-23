"""
Compile the LaTeX paper to PDF
"""

import subprocess
import os

def compile_paper():
    """Compile the LaTeX paper"""
    print("="*60)
    print("Compiling LaTeX Paper")
    print("="*60)
    
    os.chdir('obds_void_experiment')
    
    # Run pdflatex twice for references
    print("\nFirst pass...")
    result1 = subprocess.run(['pdflatex', '-interaction=nonstopmode', 
                             'bounded_latency_paper.tex'],
                            capture_output=True, text=True)
    
    if result1.returncode != 0:
        print("⚠ First pass had warnings (expected)")
    else:
        print("✓ First pass complete")
    
    # Run bibtex
    print("\nRunning BibTeX...")
    result2 = subprocess.run(['bibtex', 'bounded_latency_paper'],
                            capture_output=True, text=True)
    
    if result2.returncode != 0:
        print("⚠ BibTeX warnings (expected if no citations)")
    else:
        print("✓ BibTeX complete")
    
    # Run pdflatex again
    print("\nSecond pass...")
    result3 = subprocess.run(['pdflatex', '-interaction=nonstopmode',
                             'bounded_latency_paper.tex'],
                            capture_output=True, text=True)
    
    if result3.returncode != 0:
        print("⚠ Second pass had warnings")
    else:
        print("✓ Second pass complete")
    
    # Final pass
    print("\nFinal pass...")
    result4 = subprocess.run(['pdflatex', '-interaction=nonstopmode',
                             'bounded_latency_paper.tex'],
                            capture_output=True, text=True)
    
    if result4.returncode == 0:
        print("✓ Final pass complete")
        print("\n" + "="*60)
        print("✓ Paper compiled successfully!")
        print("="*60)
        print("\nOutput: bounded_latency_paper.pdf")
        
        # Check file size
        if os.path.exists('bounded_latency_paper.pdf'):
            size = os.path.getsize('bounded_latency_paper.pdf')
            print(f"PDF size: {size/1024:.1f} KB")
        
        return True
    else:
        print("\n✗ Compilation failed")
        print("\nLast error output:")
        print(result4.stderr[-500:] if result4.stderr else "No error output")
        return False

if __name__ == "__main__":
    success = compile_paper()
    if not success:
        print("\nNote: Manual compilation may be needed if pdflatex is not installed")
        print("Run: cd obds_void_experiment && pdflatex bounded_latency_paper.tex")
