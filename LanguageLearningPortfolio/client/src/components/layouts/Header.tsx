import { useState } from "react";
import { Link } from "wouter";
import { Menu } from "lucide-react";
import { 
  Sheet,
  SheetContent,
  SheetTrigger
} from "@/components/ui/sheet";

export default function Header() {
  const [open, setOpen] = useState(false);
  
  return (
    <header className="bg-white/80 backdrop-blur-sm shadow-sm fixed w-full z-10">
      <div className="container mx-auto px-4 py-3 flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <svg viewBox="0 0 24 24" className="w-8 h-8 rounded-full object-cover" fill="#88B2D3">
            <path d="M12,2C6.486,2,2,6.486,2,12s4.486,10,10,10s10-4.486,10-10S17.514,2,12,2z M12,20c-4.411,0-8-3.589-8-8 s3.589-8,8-8s8,3.589,8,8S16.411,20,12,20z"/>
            <path d="M13 7L11 7 11 13 17 13 17 11 13 11z"/>
          </svg>
          <h1 className="font-quicksand font-bold text-xl md:text-2xl text-[#3C4F65]">
            Ghibli<span className="text-[#88B2D3]">Transform</span>
          </h1>
        </div>
        
        <nav className="hidden md:flex items-center space-x-6">
          <Link href="/">
            <a className="font-quicksand font-medium text-[#3C4F65] hover:text-[#88B2D3] transition-colors">
              Home
            </a>
          </Link>
          <Link href="#gallery">
            <a className="font-quicksand font-medium text-[#3C4F65] hover:text-[#88B2D3] transition-colors">
              Gallery
            </a>
          </Link>
          <Link href="#features">
            <a className="font-quicksand font-medium text-[#3C4F65] hover:text-[#88B2D3] transition-colors">
              About
            </a>
          </Link>
          <Link href="#transform">
            <a className="font-quicksand font-medium text-white bg-[#88B2D3] hover:bg-[#6892B3] rounded-full px-5 py-2 transition-colors">
              Get Started
            </a>
          </Link>
        </nav>
        
        <Sheet open={open} onOpenChange={setOpen}>
          <SheetTrigger asChild>
            <button className="md:hidden text-[#3C4F65]" aria-label="Menu">
              <Menu className="h-6 w-6" />
            </button>
          </SheetTrigger>
          <SheetContent side="right">
            <div className="flex flex-col gap-6 mt-10">
              <Link href="/">
                <a 
                  className="font-quicksand font-medium text-[#3C4F65] hover:text-[#88B2D3] transition-colors"
                  onClick={() => setOpen(false)}
                >
                  Home
                </a>
              </Link>
              <Link href="#gallery">
                <a 
                  className="font-quicksand font-medium text-[#3C4F65] hover:text-[#88B2D3] transition-colors"
                  onClick={() => setOpen(false)}
                >
                  Gallery
                </a>
              </Link>
              <Link href="#features">
                <a 
                  className="font-quicksand font-medium text-[#3C4F65] hover:text-[#88B2D3] transition-colors"
                  onClick={() => setOpen(false)}
                >
                  About
                </a>
              </Link>
              <Link href="#transform">
                <a 
                  className="font-quicksand font-medium text-white bg-[#88B2D3] hover:bg-[#6892B3] rounded-full px-5 py-2 transition-colors text-center"
                  onClick={() => setOpen(false)}
                >
                  Get Started
                </a>
              </Link>
            </div>
          </SheetContent>
        </Sheet>
      </div>
    </header>
  );
}
