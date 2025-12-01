import { motion } from 'framer-motion';
import { Card } from '@/components/ui/card';
import { Shield, TrendingUp, Users, Lock, Globe, Heart } from 'lucide-react';
import { useScrollAnimation } from '@/hooks/useScrollAnimation';

const values = [
    { icon: Heart, title: 'Commitment', description: 'Dedicated to cooperative success' },
    { icon: Shield, title: 'Transparency', description: 'Real-time visibility & compliance' },
    { icon: TrendingUp, title: 'Affordability', description: 'Pay-as-you-use subscription' },
    { icon: Lock, title: 'Security', description: 'Encrypted, audit-ready protection' },
    { icon: Globe, title: 'Innovation', description: 'AI-powered Analytics & Cloud' },
    { icon: Users, title: 'Inclusion', description: 'Digital enablement for all' },
];

export const About = () => {
    const { ref, isVisible } = useScrollAnimation();

    return (
        <section id="about" className="relative py-24 overflow-hidden">
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
                        About <span className="gradient-text">CreditIQ</span>
                    </h2>
                    <p className="text-xl text-gray-300 max-w-3xl mx-auto">
                        We are a dedicated SaaS technology company transforming the cooperative sector of India
                        through digital innovation, automation, transparency and data-driven decision systems.
                    </p>
                </motion.div>

                {/* Mission */}
                <motion.div
                    initial={{ opacity: 0, y: 30 }}
                    animate={isVisible ? { opacity: 1, y: 0 } : {}}
                    transition={{ duration: 0.8, delay: 0.2 }}
                    className="max-w-4xl mx-auto mb-16"
                >
                    <Card className="text-center p-8 md:p-12 relative overflow-hidden">
                        <div className="absolute inset-0 bg-gradient-to-br from-blue-500/10 to-purple-500/10"></div>
                        <div className="relative z-10">
                            <h3 className="text-2xl md:text-3xl font-heading font-semibold mb-4 gradient-text">
                                Our Purpose
                            </h3>
                            <p className="text-lg text-gray-300">
                                To empower cooperatives with modern technology that simplifies operations,
                                improves member services, and accelerates financial and economic growth.
                            </p>
                        </div>
                    </Card>
                </motion.div>

                {/* Values Grid */}
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 max-w-6xl mx-auto">
                    {values.map((value, index) => {
                        const Icon = value.icon;
                        return (
                            <motion.div
                                key={index}
                                initial={{ opacity: 0, y: 30 }}
                                animate={isVisible ? { opacity: 1, y: 0 } : {}}
                                transition={{ duration: 0.6, delay: 0.3 + index * 0.1 }}
                            >
                                <Card className="h-full p-6 group hover:border-primary-500/50 transition-all duration-300">
                                    <motion.div
                                        whileHover={{ scale: 1.1, rotate: 5 }}
                                        transition={{ type: "spring", stiffness: 300 }}
                                        className="w-12 h-12 rounded-lg bg-gradient-to-br from-primary-500 to-accent-purple-500 flex items-center justify-center mb-4"
                                    >
                                        <Icon className="w-6 h-6 text-white" />
                                    </motion.div>
                                    <h4 className="text-xl font-heading font-semibold mb-2 group-hover:text-primary-400 transition-colors">
                                        {value.title}
                                    </h4>
                                    <p className="text-gray-400">{value.description}</p>
                                </Card>
                            </motion.div>
                        );
                    })}
                </div>

                {/* Deep Understanding */}
                <motion.div
                    initial={{ opacity: 0, y: 30 }}
                    animate={isVisible ? { opacity: 1, y: 0 } : {}}
                    transition={{ duration: 0.8, delay: 0.8 }}
                    className="mt-16 text-center max-w-3xl mx-auto"
                >
                    <p className="text-lg text-gray-300 leading-relaxed">
                        With a deep understanding of cooperative workflows — from grassroots PACS to apex federations —
                        we deliver <span className="text-primary-400 font-semibold">secure</span>,
                        <span className="text-accent-purple-500 font-semibold"> scalable</span> and
                        <span className="text-primary-400 font-semibold"> field-ready</span> platforms
                        tailored for the needs of rural institutions.
                    </p>
                </motion.div>
            </div>
        </section>
    );
};
