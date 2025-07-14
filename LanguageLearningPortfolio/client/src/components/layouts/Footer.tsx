import { Link } from "wouter";
import { 
  FacebookIcon, 
  TwitterIcon, 
  InstagramIcon, 
  Facebook,
  Send
} from "lucide-react";

export default function Footer() {
  return (
    <footer className="bg-[#3C4F65] py-12 text-white/80">
      <div className="container mx-auto px-4">
        <div className="max-w-6xl mx-auto">
          {/* Top Footer */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-8 mb-12">
            {/* Brand Column */}
            <div className="md:col-span-1">
              <div className="flex items-center space-x-2 mb-4">
                <svg viewBox="0 0 24 24" className="w-8 h-8 rounded-full object-cover" fill="#88B2D3">
                  <path d="M12,2C6.486,2,2,6.486,2,12s4.486,10,10,10s10-4.486,10-10S17.514,2,12,2z M12,20c-4.411,0-8-3.589-8-8 s3.589-8,8-8s8,3.589,8,8S16.411,20,12,20z"/>
                  <path d="M13 7L11 7 11 13 17 13 17 11 13 11z"/>
                </svg>
                <h3 className="font-quicksand font-bold text-xl text-white">
                  Ghibli<span className="text-[#88B2D3]">Transform</span>
                </h3>
              </div>
              <p className="mb-4">Transform your photos into magical Studio Ghibli inspired artwork using the power of AI.</p>
              <div className="flex space-x-4">
                <a href="#" className="text-white/70 hover:text-[#88B2D3] transition-colors">
                  <InstagramIcon className="h-5 w-5" />
                </a>
                <a href="#" className="text-white/70 hover:text-[#88B2D3] transition-colors">
                  <TwitterIcon className="h-5 w-5" />
                </a>
                <a href="#" className="text-white/70 hover:text-[#88B2D3] transition-colors">
                  <FacebookIcon className="h-5 w-5" />
                </a>
                <a href="#" className="text-white/70 hover:text-[#88B2D3] transition-colors">
                  <Facebook className="h-5 w-5" />
                </a>
              </div>
            </div>
            
            {/* Quick Links */}
            <div>
              <h4 className="font-quicksand font-semibold text-white mb-4">Quick Links</h4>
              <ul className="space-y-2">
                <li>
                  <Link href="/">
                    <a className="hover:text-[#88B2D3] transition-colors">Home</a>
                  </Link>
                </li>
                <li>
                  <Link href="#gallery">
                    <a className="hover:text-[#88B2D3] transition-colors">Gallery</a>
                  </Link>
                </li>
                <li>
                  <Link href="#features">
                    <a className="hover:text-[#88B2D3] transition-colors">How It Works</a>
                  </Link>
                </li>
                <li>
                  <Link href="#transform">
                    <a className="hover:text-[#88B2D3] transition-colors">Transform</a>
                  </Link>
                </li>
                <li>
                  <Link href="#">
                    <a className="hover:text-[#88B2D3] transition-colors">Contact Us</a>
                  </Link>
                </li>
              </ul>
            </div>
            
            {/* Legal */}
            <div>
              <h4 className="font-quicksand font-semibold text-white mb-4">Legal</h4>
              <ul className="space-y-2">
                <li>
                  <Link href="#">
                    <a className="hover:text-[#88B2D3] transition-colors">Terms of Service</a>
                  </Link>
                </li>
                <li>
                  <Link href="#">
                    <a className="hover:text-[#88B2D3] transition-colors">Privacy Policy</a>
                  </Link>
                </li>
                <li>
                  <Link href="#">
                    <a className="hover:text-[#88B2D3] transition-colors">Copyright</a>
                  </Link>
                </li>
                <li>
                  <Link href="#">
                    <a className="hover:text-[#88B2D3] transition-colors">Cookies</a>
                  </Link>
                </li>
                <li>
                  <Link href="#">
                    <a className="hover:text-[#88B2D3] transition-colors">GDPR</a>
                  </Link>
                </li>
              </ul>
            </div>
            
            {/* Newsletter */}
            <div>
              <h4 className="font-quicksand font-semibold text-white mb-4">Stay Updated</h4>
              <p className="mb-4">Subscribe to our newsletter for updates and new features.</p>
              <form className="flex">
                <input 
                  type="email" 
                  placeholder="Your email" 
                  className="rounded-l-lg px-4 py-2 w-full text-[#3C4F65] outline-none focus:ring-2 focus:ring-[#88B2D3]" 
                />
                <button 
                  type="submit" 
                  className="bg-[#88B2D3] hover:bg-[#6892B3] transition-colors rounded-r-lg px-4 flex items-center justify-center"
                >
                  <Send className="h-4 w-4" />
                </button>
              </form>
            </div>
          </div>
          
          {/* Bottom Footer */}
          <div className="pt-6 border-t border-white/10 flex flex-col md:flex-row justify-between items-center">
            <p className="text-sm">© {new Date().getFullYear()} GhibliTransform. All rights reserved.</p>
            <p className="text-sm mt-2 md:mt-0">
              Made with <span className="text-[#F4A6A1]">♥</span> for Studio Ghibli fans everywhere
            </p>
          </div>
        </div>
      </div>
    </footer>
  );
}
