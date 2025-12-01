import { motion } from 'framer-motion';
import { Check, Zap, Cloud, Shield, Globe2, Languages } from 'lucide-react';
import { useScrollAnimation } from '@/hooks/useScrollAnimation';

const features = [
    {
        icon: Zap,
        title: 'Designed EXCLUSIVELY for Indian Cooperatives',
        description: 'Built from ground up for cooperative workflows',
    },
    {
        icon: Cloud,
        title: 'No IT Infrastructure — just subscribe & start',
        description: 'Cloud-based SaaS with zero setup hassle',
    },
    {
        icon: Check,
        title: 'Fast deployment & modular implementation',
        description: 'Get started in days, not months',
    },
    {
        icon: Shield,
        title: 'Data security with 24×7 cloud backup',
        description: 'Enterprise-grade security and reliability',
    },
    {
        icon: Globe2,
        title: 'Integrations with UIDAI, NPCI, CBS, PM-Schemes',
        description: 'Seamless connectivity with government systems',
    },
    {
        icon: Languages,
        title: 'Support in English + Hindi + Regional Interfaces',
        description: 'Multilingual support for all users',
    },
];

export const WhyChooseUs = () => {
    const { ref, isVisible } = useScrollAnimation();

    return (
        <section id="why-choose-us" className="relative py-24 overflow-hidden">
            {/* Background */}
            <div className="absolute inset-0 bg-gradient-to-b from-purple-950/50 to-slate-950"></div>

            <div className="relative z-10 container mx-auto px-4">
                <motion.div
                    ref={ref}
                    initial={{ opacity: 0, y: 50 }}
                    animate={isVisible ? { opacity: 1, y: 0 } : {}}
                    transition={{ duration: 0.8 }}
                    className="text-center mb-16"
                >
                    <h2 className="text-4xl md:text-6xl font-heading font-bold mb-6">
                        Why Choose <span className="gradient-text">CreditIQ</span>
                    </h2>
                    <p className="text-xl text-gray-300 max-w-3xl mx-auto">
                        Built for cooperatives, by experts who understand your unique challenges
                    </p>
                </motion.div>

                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8 max-w-6xl mx-auto">
                    {features.map((feature, index) => {
                        const Icon = feature.icon;
                        return (
                            <motion.div
                                key={index}
                                initial={{ opacity: 0, x: -30 }}
                                animate={isVisible ? { opacity: 1, x: 0 } : {}}
                                transition={{ duration: 0.6, delay: index * 0.1 }}
                                className="flex gap-4 group"
                            >
                                <div className="flex-shrink-0">
                                    <motion.div
                                        whileHover={{ scale: 1.2, rotate: 10 }}
                                        className="w-12 h-12 rounded-lg bg-gradient-to-br from-primary-500 to-accent-purple-500 flex items-center justify-center group-hover:shadow-lg group-hover:shadow-primary-500/50 transition-shadow"
                                    >
                                        <Icon className="w-6 h-6 text-white" />
                                    </motion.div>
                                </div>
                                <div className="flex-1">
                                    <h3 className="text-lg font-semibold mb-2 group-hover:text-primary-400 transition-colors">
                                        {feature.title}
                                    </h3>
                                    <p className="text-gray-400 text-sm leading-relaxed">
                                        {feature.description}
                                    </p>
                                </div>
                            </motion.div>
                        );
                    })}
                </div>

                {/* Additional highlight */}
                <motion.div
                    initial={{ opacity: 0, y: 30 }}
                    animate={isVisible ? { opacity: 1, y: 0 } : {}}
                    transition={{ duration: 0.8, delay: 0.8 }}
                    className="mt-16 text-center"
                >
                    <div className="inline-block glass-card px-8 py-6 rounded-2xl">
                        <p className="text-lg md:text-xl">
                            <span className="gradient-text font-semibold">Trusted by 500+ cooperatives</span> across India
                        </p>
                    </div>
                </motion.div>
            </div>
        </section>
    );
};
