import { useState } from "react";
import { Link } from "wouter";
import { 
  Sheet, 
  SheetContent, 
  SheetTrigger, 
  SheetClose 
} from "@/components/ui/sheet";
import { Dumbbell, Menu } from "lucide-react";

const Header = () => {
  const [isMenuOpen, setIsMenuOpen] = useState(false);

  const handleCloseMenu = () => {
    setIsMenuOpen(false);
  };

  const navigationLinks = [
    { href: "/#bmi-calculator", label: "BMI Calculator" },
    { href: "/#body-types", label: "Body Types" },
    { href: "/#workouts", label: "Workouts" },
    { href: "/#nutrition", label: "Nutrition" },
    { href: "/#education", label: "Education" },
  ];

  return (
    <header className="bg-primary text-white shadow-md">
      <div className="container mx-auto px-4 py-3 flex justify-between items-center">
        <div className="flex items-center space-x-2">
          <Dumbbell className="h-6 w-6" />
          <Link href="/">
            <a className="font-heading font-bold text-xl md:text-2xl">FitTrack</a>
          </Link>
        </div>
        
        {/* Desktop Navigation */}
        <nav className="hidden md:block">
          <ul className="flex space-x-6">
            {navigationLinks.map((link) => (
              <li key={link.label}>
                <a 
                  href={link.href} 
                  className="hover:text-amber-300 transition"
                  onClick={(e) => {
                    // If we're on the home page, handle anchor links with smooth scrolling
                    if (window.location.pathname === '/' && link.href.startsWith('#')) {
                      e.preventDefault();
                      const targetId = link.href.substring(2); // Remove '/#'
                      const element = document.getElementById(targetId);
                      if (element) {
                        element.scrollIntoView({ behavior: 'smooth' });
                      }
                    }
                  }}
                >
                  {link.label}
                </a>
              </li>
            ))}
          </ul>
        </nav>
        
        {/* Mobile Menu Button */}
        <Sheet open={isMenuOpen} onOpenChange={setIsMenuOpen}>
          <SheetTrigger asChild>
            <button className="md:hidden text-xl">
              <Menu />
            </button>
          </SheetTrigger>
          <SheetContent side="right" className="bg-primary text-white p-6 w-[250px]">
            <nav>
              <ul className="space-y-4 mt-8">
                {navigationLinks.map((link) => (
                  <li key={link.label}>
                    <SheetClose asChild>
                      <a 
                        href={link.href} 
                        className="block py-2 hover:text-amber-300 transition"
                        onClick={(e) => {
                          handleCloseMenu();
                          // If we're on the home page, handle anchor links with smooth scrolling
                          if (window.location.pathname === '/' && link.href.startsWith('#')) {
                            e.preventDefault();
                            const targetId = link.href.substring(2); // Remove '/#'
                            const element = document.getElementById(targetId);
                            if (element) {
                              element.scrollIntoView({ behavior: 'smooth' });
                            }
                          }
                        }}
                      >
                        {link.label}
                      </a>
                    </SheetClose>
                  </li>
                ))}
              </ul>
            </nav>
          </SheetContent>
        </Sheet>
      </div>
    </header>
  );
};

export default Header;
