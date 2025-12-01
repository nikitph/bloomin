import { motion } from 'framer-motion';
import { Card } from '@/components/ui/card';
import { AnimatedCounter } from '@/components/AnimatedCounter';
import { TrendingUp, Users, Shield, Zap } from 'lucide-react';
import { useScrollAnimation } from '@/hooks/useScrollAnimation';

const impacts = [
    {
        icon: Zap,
        title: 'Automation',
        description: 'Reduced manual errors & paperwork',
        metric: '90',
        suffix: '%',
        color: 'from-yellow-500 to-orange-500',
    },
    {
        icon: Shield,
        title: 'Transparency',
        description: 'Trustworthy governance & audits',
        metric: '100',
        suffix: '%',
        color: 'from-blue-500 to-cyan-500',
    },
    {
        icon: TrendingUp,
        title: 'Real-time MIS',
        description: 'Faster policy & lending decisions',
        metric: '75',
        suffix: '%',
        color: 'from-green-500 to-emerald-500',
    },
    {
        icon: Users,
        title: 'Member Satisfaction',
        description: 'Improved satisfaction & retention',
        metric: '95',
        suffix: '%',
        color: 'from-purple-500 to-pink-500',
    },
];

export const Impact = () => {
    const { ref, isVisible } = useScrollAnimation();

    return (
        <section id="impact" className="relative py-24 overflow-hidden">
            {/* Background */}
            <div className="absolute inset-0 bg-gradient-to-b from-slate-950 to-blue-950/50"></div>

            <div className="relative z-10 container mx-auto px-4">
                <motion.div
                    ref={ref}
                    initial={{ opacity: 0, y: 50 }}
                    animate={isVisible ? { opacity: 1, y: 0 } : {}}
                    transition={{ duration: 0.8 }}
                    className="text-center mb-16"
                >
                    <h2 className="text-4xl md:text-6xl font-heading font-bold mb-6">
                        Impact We <span className="gradient-text">Create</span>
                    </h2>
                    <p className="text-xl text-gray-300 max-w-3xl mx-auto">
                        Measurable improvements that transform cooperative operations
                    </p>
                </motion.div>

                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 max-w-7xl mx-auto">
                    {impacts.map((impact, index) => {
                        const Icon = impact.icon;
                        return (
                            <motion.div
                                key={index}
                                initial={{ opacity: 0, scale: 0.8 }}
                                animate={isVisible ? { opacity: 1, scale: 1 } : {}}
                                transition={{ duration: 0.6, delay: index * 0.15 }}
                            >
                                <Card className="h-full text-center p-6 group hover:scale-105 transition-transform">
                                    <motion.div
                                        whileHover={{ rotate: 360 }}
                                        transition={{ duration: 0.8 }}
                                        className={`w-16 h-16 mx-auto rounded-full bg-gradient-to-br ${impact.color} flex items-center justify-center mb-4`}
                                    >
                                        <Icon className="w-8 h-8 text-white" />
                                    </motion.div>

                                    <div className="text-5xl font-bold mb-2">
                                        <AnimatedCounter end={parseInt(impact.metric)} suffix={impact.suffix} />
                                    </div>

                                    <h3 className="text-xl font-semibold mb-2 group-hover:text-primary-400 transition-colors">
                                        {impact.title}
                                    </h3>

                                    <p className="text-gray-400 text-sm">
                                        {impact.description}
                                    </p>
                                </Card>
                            </motion.div>
                        );
                    })}
                </div>

                {/* Additional Benefits */}
                <motion.div
                    initial={{ opacity: 0, y: 30 }}
                    animate={isVisible ? { opacity: 1, y: 0 } : {}}
                    transition={{ duration: 0.8, delay: 0.8 }}
                    className="mt-16 max-w-4xl mx-auto"
                >
                    <Card className="p-8">
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                            <div className="text-center">
                                <div className="text-4xl font-bold gradient-text mb-2">
                                    <AnimatedCounter end={500} suffix="+" />
                                </div>
                                <p className="text-gray-400">Active Cooperatives</p>
                            </div>
                            <div className="text-center">
                                <div className="text-4xl font-bold gradient-text mb-2">
                                    <AnimatedCounter end={1000000} suffix="+" />
                                </div>
                                <p className="text-gray-400">Members Served</p>
                            </div>
                        </div>
                    </Card>
                </motion.div>
            </div>
        </section>
    );
};
