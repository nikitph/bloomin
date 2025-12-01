import { Navbar } from './components/Navbar';
import { Hero } from './sections/Hero';
import { About } from './sections/About';
import { Solutions } from './sections/Solutions';
import { WhyChooseUs } from './sections/WhyChooseUs';
import { Impact } from './sections/Impact';
import { Contact } from './sections/Contact';
import './index.css';

function App() {
  return (
    <div className="min-h-screen bg-slate-950 text-white">
      <Navbar />
      <main>
        <div id="home">
          <Hero />
        </div>
        <About />
        <Solutions />
        <WhyChooseUs />
        <Impact />
        <Contact />
      </main>
    </div>
  );
}

export default App;
