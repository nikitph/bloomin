import { motion } from 'framer-motion';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import {
    BarChart3,
    Star,
    LayoutDashboard,
    Wallet,
    FileCheck,
    Search,
    ClipboardCheck,
    Award
} from 'lucide-react';
import { useScrollAnimation } from '@/hooks/useScrollAnimation';

const solutions = [
    {
        icon: BarChart3,
        title: 'PACS Business Diversification Suite',
        description: 'Comprehensive tools to expand revenue streams and service offerings for PACS',
        gradient: 'from-blue-500 to-cyan-500',
    },
    {
        icon: Star,
        title: 'PACS Rating Model',
        description: 'Advanced rating system to assess and improve PACS performance metrics',
        gradient: 'from-purple-500 to-pink-500',
    },
    {
        icon: LayoutDashboard,
        title: 'Governance & MIS Dashboard',
        description: 'Real-time, multi-level insights for Board, Management & Regulators with State → District → PACS layered view',
        gradient: 'from-orange-500 to-red-500',
    },
    {
        icon: Wallet,
        title: 'Loan Operating System',
        description: 'Complete loan management system for cooperative banks with automated workflows',
        gradient: 'from-green-500 to-emerald-500',
    },
    {
        icon: FileCheck,
        title: 'Online Concurrent Audit',
        description: 'Real-time audit capabilities with automated compliance checking',
        gradient: 'from-indigo-500 to-blue-500',
    },
    {
        icon: Search,
        title: 'Online Information System Audit',
        description: 'Comprehensive IS audit tools for data integrity and security verification',
        gradient: 'from-violet-500 to-purple-500',
    },
    {
        icon: ClipboardCheck,
        title: 'Online Statutory Audit',
        description: 'Digital statutory audit platform with regulatory compliance tracking',
        gradient: 'from-pink-500 to-rose-500',
    },
    {
        icon: Award,
        title: 'Online Inspection and Rating',
        description: 'Automated inspection workflows with performance rating algorithms',
        gradient: 'from-teal-500 to-cyan-500',
    },
];

export const Solutions = () => {
    const { ref, isVisible } = useScrollAnimation();

    return (
        <section id="solutions" className="relative py-24 overflow-hidden">
            {/* Background */}
            <div className="absolute inset-0 bg-gradient-to-b from-blue-950/50 to-purple-950/50"></div>

            <div className="relative z-10 container mx-auto px-4">
                <motion.div
                    ref={ref}
                    initial={{ opacity: 0, y: 50 }}
                    animate={isVisible ? { opacity: 1, y: 0 } : {}}
                    transition={{ duration: 0.8 }}
                    className="text-center mb-16"
                >
                    <h2 className="text-4xl md:text-6xl font-heading font-bold mb-6">
                        Our <span className="gradient-text">Solutions</span>
                    </h2>
                    <p className="text-xl text-gray-300 max-w-3xl mx-auto">
                        Comprehensive digital solutions designed specifically for the cooperative sector
                    </p>
                </motion.div>

                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 max-w-7xl mx-auto">
                    {solutions.map((solution, index) => {
                        const Icon = solution.icon;
                        return (
                            <motion.div
                                key={index}
                                initial={{ opacity: 0, scale: 0.9 }}
                                animate={isVisible ? { opacity: 1, scale: 1 } : {}}
                                transition={{ duration: 0.5, delay: index * 0.1 }}
                                whileHover={{ y: -8 }}
                            >
                                <Card className="h-full group cursor-pointer relative overflow-hidden">
                                    {/* Gradient overlay on hover */}
                                    <div className={`absolute inset-0 bg-gradient-to-br ${solution.gradient} opacity-0 group-hover:opacity-10 transition-opacity duration-300`}></div>

                                    <CardHeader>
                                        <motion.div
                                            whileHover={{ rotate: 360, scale: 1.1 }}
                                            transition={{ duration: 0.6 }}
                                            className={`w-14 h-14 rounded-xl bg-gradient-to-br ${solution.gradient} flex items-center justify-center mb-4`}
                                        >
                                            <Icon className="w-7 h-7 text-white" />
                                        </motion.div>
                                        <CardTitle className="text-xl group-hover:text-primary-400 transition-colors">
                                            {solution.title}
                                        </CardTitle>
                                    </CardHeader>
                                    <CardContent>
                                        <p className="text-gray-400 leading-relaxed">
                                            {solution.description}
                                        </p>
                                    </CardContent>

                                    {/* Animated border */}
                                    <div className="absolute inset-0 rounded-xl opacity-0 group-hover:opacity-100 transition-opacity duration-300">
                                        <div className={`absolute inset-0 rounded-xl bg-gradient-to-r ${solution.gradient} blur-xl opacity-30`}></div>
                                    </div>
                                </Card>
                            </motion.div>
                        );
                    })}
                </div>
            </div>
        </section>
    );
};
