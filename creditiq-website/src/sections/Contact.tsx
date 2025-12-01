import { motion } from 'framer-motion';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Mail, Phone, MapPin, Calendar, ArrowRight } from 'lucide-react';
import { useScrollAnimation } from '@/hooks/useScrollAnimation';

export const Contact = () => {
    const { ref, isVisible } = useScrollAnimation();

    return (
        <section id="contact" className="relative py-24 overflow-hidden">
            {/* Background */}
            <div className="absolute inset-0 bg-gradient-to-b from-blue-950/50 to-slate-950"></div>

            <div className="relative z-10 container mx-auto px-4">
                <motion.div
                    ref={ref}
                    initial={{ opacity: 0, y: 50 }}
                    animate={isVisible ? { opacity: 1, y: 0 } : {}}
                    transition={{ duration: 0.8 }}
                    className="text-center mb-16"
                >
                    <h2 className="text-4xl md:text-6xl font-heading font-bold mb-6">
                        Get in <span className="gradient-text">Touch</span>
                    </h2>
                    <p className="text-xl text-gray-300 max-w-3xl mx-auto">
                        We're ready to help you digitize your cooperative
                    </p>
                </motion.div>

                <div className="max-w-5xl mx-auto">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-12">
                        {[
                            { icon: Mail, title: 'Email Us', value: 'contact@creditiq.in', href: 'mailto:contact@creditiq.in' },
                            { icon: Phone, title: 'Call Us', value: '+91 XXXX XXXXXX', href: 'tel:+91XXXXXXXXXX' },
                            { icon: MapPin, title: 'Visit Us', value: 'Mumbai, Maharashtra, India', href: '#' },
                            { icon: Calendar, title: 'Book Demo', value: 'Schedule a consultation', href: '#' },
                        ].map((contact, index) => {
                            const Icon = contact.icon;
                            return (
                                <motion.div
                                    key={index}
                                    initial={{ opacity: 0, y: 30 }}
                                    animate={isVisible ? { opacity: 1, y: 0 } : {}}
                                    transition={{ duration: 0.6, delay: index * 0.1 }}
                                >
                                    <Card className="p-6 group hover:border-primary-500/50 transition-all cursor-pointer">
                                        <div className="flex items-start gap-4">
                                            <motion.div
                                                whileHover={{ scale: 1.1, rotate: 5 }}
                                                className="w-12 h-12 rounded-lg bg-gradient-to-br from-primary-500 to-accent-purple-500 flex items-center justify-center flex-shrink-0"
                                            >
                                                <Icon className="w-6 h-6 text-white" />
                                            </motion.div>
                                            <div className="flex-1">
                                                <h3 className="text-lg font-semibold mb-1 group-hover:text-primary-400 transition-colors">
                                                    {contact.title}
                                                </h3>
                                                <p className="text-gray-400">{contact.value}</p>
                                            </div>
                                        </div>
                                    </Card>
                                </motion.div>
                            );
                        })}
                    </div>

                    {/* CTA Section */}
                    <motion.div
                        initial={{ opacity: 0, y: 30 }}
                        animate={isVisible ? { opacity: 1, y: 0 } : {}}
                        transition={{ duration: 0.8, delay: 0.5 }}
                        className="text-center"
                    >
                        <Card className="p-12 relative overflow-hidden">
                            <div className="absolute inset-0 bg-gradient-to-br from-primary-500/10 to-accent-purple-500/10"></div>
                            <div className="relative z-10">
                                <h3 className="text-3xl font-heading font-bold mb-4">
                                    Ready to Transform Your Cooperative?
                                </h3>
                                <p className="text-gray-300 mb-8 max-w-2xl mx-auto">
                                    Join hundreds of cooperatives already benefiting from our digital solutions.
                                    Schedule a demo today and see the difference.
                                </p>
                                <div className="flex flex-col sm:flex-row gap-4 justify-center">
                                    <Button variant="gradient" size="xl" className="group">
                                        Request Demo
                                        <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
                                    </Button>
                                    <Button variant="outline" size="xl" className="glass hover:bg-white/10">
                                        Download Brochure
                                    </Button>
                                </div>
                            </div>
                        </Card>
                    </motion.div>
                </div>
            </div>

            {/* Footer */}
            <motion.div
                initial={{ opacity: 0 }}
                animate={isVisible ? { opacity: 1 } : {}}
                transition={{ duration: 0.8, delay: 0.8 }}
                className="relative z-10 container mx-auto px-4 mt-20 pt-8 border-t border-white/10"
            >
                <div className="text-center text-gray-400 text-sm">
                    <p>© 2024 CreditIQ. All rights reserved.</p>
                    <p className="mt-2">Empowering India's Cooperatives with Next-Gen Technology</p>
                </div>
            </motion.div>
        </section>
    );
};
