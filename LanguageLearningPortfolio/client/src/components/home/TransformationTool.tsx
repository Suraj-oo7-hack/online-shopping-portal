import { useState } from "react";
import { useToast } from "@/hooks/use-toast";
import { useMutation } from "@tanstack/react-query";
import { apiRequest } from "@/lib/queryClient";
import { Dropzone } from "@/components/ui/dropzone";
import { ComparisonSlider } from "@/components/ui/comparison-slider";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Slider } from "@/components/ui/slider";
import { Button } from "@/components/ui/button";
import { 
  DownloadIcon, 
  Share2Icon, 
  WandSparklesIcon, 
  RotateCcwIcon,
  Sliders 
} from "lucide-react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

export default function TransformationTool() {
  const { toast } = useToast();
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [styleIntensity, setStyleIntensity] = useState(75);
  const [selectedStyle, setSelectedStyle] = useState("totoro");
  const [selectedQuality, setSelectedQuality] = useState("standard");
  const [selectedPalette, setSelectedPalette] = useState("default");
  const [originalImageUrl, setOriginalImageUrl] = useState<string | null>(null);
  const [transformedImageUrl, setTransformedImageUrl] = useState<string | null>(null);
  
  const resetForm = () => {
    setSelectedFile(null);
    setStyleIntensity(75);
    setSelectedStyle("totoro");
    setSelectedQuality("standard");
    setSelectedPalette("default");
    setOriginalImageUrl(null);
    setTransformedImageUrl(null);
  };
  
  const handleImageSelect = (file: File) => {
    setSelectedFile(file);
    const imageUrl = URL.createObjectURL(file);
    setOriginalImageUrl(imageUrl);
    setTransformedImageUrl(null);
  };
  
  const transformMutation = useMutation({
    mutationFn: async () => {
      if (!selectedFile) throw new Error("No image selected");
      
      const formData = new FormData();
      formData.append("image", selectedFile);
      formData.append("style", selectedStyle);
      formData.append("intensity", styleIntensity.toString());
      formData.append("quality", selectedQuality);
      formData.append("palette", selectedPalette);
      
      const response = await fetch("/api/transform", {
        method: "POST",
        body: formData,
        credentials: "include",
      });
      
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(errorText || "Failed to transform image");
      }
      
      const result = await response.json();
      return result;
    },
    onSuccess: (data) => {
      setTransformedImageUrl(data.transformedImageUrl);
      toast({
        title: "Transformation complete!",
        description: "Your image has been successfully transformed.",
      });
    },
    onError: (error) => {
      toast({
        title: "Transformation failed",
        description: error.message || "Something went wrong. Please try again.",
        variant: "destructive",
      });
    },
  });
  
  const handleTransform = () => {
    if (!selectedFile) {
      toast({
        title: "No image selected",
        description: "Please upload an image to transform.",
        variant: "destructive",
      });
      return;
    }
    
    transformMutation.mutate();
  };
  
  const handleDownload = () => {
    if (!transformedImageUrl) return;
    
    const link = document.createElement("a");
    link.href = transformedImageUrl;
    link.download = `ghibli-transform-${Date.now()}.png`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };
  
  const colorPalettes = [
    { id: "default", color: "#88B2D3", title: "Sky Blue" },
    { id: "green", color: "#A7C957", title: "Leaf Green" },
    { id: "pink", color: "#F4A6A1", title: "Sunset Pink" },
    { id: "yellow", color: "#FFD166", title: "Warm Yellow" },
    { id: "custom", gradient: "from-[#88B2D3] to-[#F4A6A1]", title: "Custom" }
  ];
  
  return (
    <section id="transform" className="py-16 md:py-24 bg-white relative">
      <div className="container mx-auto px-4">
        <div className="max-w-6xl mx-auto">
          <h2 className="font-quicksand font-bold text-2xl md:text-4xl text-[#3C4F65] text-center mb-12">
            Transform Your Images
            <div className="w-20 h-1 bg-[#88B2D3] mx-auto mt-4 rounded-full"></div>
          </h2>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 md:gap-12">
            {/* Upload & Controls Panel */}
            <div className="bg-[#F7F3E9]/50 rounded-3xl shadow-md p-6 md:p-8">
              <Dropzone onImageSelect={handleImageSelect} />

              <div>
                <h3 className="font-quicksand font-semibold text-lg text-[#3C4F65] mb-4">Transformation Settings</h3>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                  {/* Style Selector */}
                  <div>
                    <label className="block font-medium text-[#3C4F65] mb-2">Ghibli Style</label>
                    <Select value={selectedStyle} onValueChange={setSelectedStyle}>
                      <SelectTrigger className="w-full bg-white border border-gray-200 rounded-xl px-4 py-2.5 appearance-none">
                        <SelectValue placeholder="Select a style" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="totoro">My Neighbor Totoro</SelectItem>
                        <SelectItem value="spirited">Spirited Away</SelectItem>
                        <SelectItem value="mononoke">Princess Mononoke</SelectItem>
                        <SelectItem value="howl">Howl's Moving Castle</SelectItem>
                        <SelectItem value="kiki">Kiki's Delivery Service</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  
                  {/* Intensity Slider */}
                  <div>
                    <label className="block font-medium text-[#3C4F65] mb-2">Style Intensity</label>
                    <div className="relative">
                      <Slider
                        value={[styleIntensity]}
                        min={0}
                        max={100}
                        step={1}
                        onValueChange={(values) => setStyleIntensity(values[0])}
                      />
                      <div className="flex justify-between text-xs text-[#3C4F65]/60 mt-1">
                        <span>Subtle</span>
                        <span>Strong</span>
                      </div>
                    </div>
                  </div>
                </div>
                
                {/* Advanced Options */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
                  {/* Color Balance */}
                  <div>
                    <label className="block font-medium text-[#3C4F65] mb-2">Color Palette</label>
                    <div className="flex items-center space-x-3">
                      {colorPalettes.map((palette) => (
                        <button
                          key={palette.id}
                          className={`w-8 h-8 rounded-full border-2 cursor-pointer shadow-sm ${
                            selectedPalette === palette.id
                              ? "border-[#3C4F65]"
                              : "border-white"
                          } ${
                            palette.gradient
                              ? `bg-gradient-to-r ${palette.gradient}`
                              : ""
                          }`}
                          style={palette.color ? { backgroundColor: palette.color } : {}}
                          title={palette.title}
                          onClick={() => setSelectedPalette(palette.id)}
                        />
                      ))}
                    </div>
                  </div>
                  
                  {/* Image Quality */}
                  <div>
                    <label className="block font-medium text-[#3C4F65] mb-2">Output Quality</label>
                    <Select value={selectedQuality} onValueChange={setSelectedQuality}>
                      <SelectTrigger className="w-full bg-white border border-gray-200 rounded-xl px-4 py-2.5 appearance-none">
                        <SelectValue placeholder="Select quality" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="standard">Standard (1024×1024)</SelectItem>
                        <SelectItem value="high">High (2048×2048)</SelectItem>
                        <SelectItem value="ultra">Ultra (4096×4096)</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>
                
                {/* Action Buttons */}
                <div className="flex flex-col sm:flex-row gap-3 mt-6">
                  <Button
                    className="font-quicksand font-semibold bg-[#88B2D3] hover:bg-[#6892B3] text-white rounded-xl px-6 py-3 transition-all shadow-sm flex-1"
                    onClick={handleTransform}
                    disabled={!selectedFile || transformMutation.isPending}
                  >
                    {transformMutation.isPending ? (
                      <>
                        <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                        </svg>
                        Transforming...
                      </>
                    ) : (
                      <>
                        <WandSparklesIcon className="mr-2 h-4 w-4" />
                        Transform Image
                      </>
                    )}
                  </Button>
                  <Button
                    variant="outline"
                    className="font-quicksand font-semibold border border-gray-300 bg-white hover:bg-gray-50 text-[#3C4F65] rounded-xl px-6 py-3 transition-all flex-1"
                    onClick={resetForm}
                  >
                    <RotateCcwIcon className="mr-2 h-4 w-4" />
                    Reset
                  </Button>
                </div>
              </div>
            </div>
            
            {/* Preview Panel */}
            <div className="bg-white rounded-3xl shadow-md overflow-hidden">
              {/* Preview Tabs */}
              <Tabs defaultValue="preview">
                <TabsList className="flex w-full border-b">
                  <TabsTrigger 
                    value="preview" 
                    className="flex-1 py-4 px-6 font-quicksand font-medium data-[state=active]:text-[#88B2D3] data-[state=active]:border-b-2 data-[state=active]:border-[#88B2D3]"
                  >
                    <Sliders className="mr-2 h-4 w-4" />
                    Before & After
                  </TabsTrigger>
                  <TabsTrigger 
                    value="download" 
                    className="flex-1 py-4 px-6 font-quicksand font-medium"
                    disabled={!transformedImageUrl}
                  >
                    <DownloadIcon className="mr-2 h-4 w-4" />
                    Download
                  </TabsTrigger>
                </TabsList>
                <TabsContent value="preview" className="p-6 flex flex-col items-center">
                  <div className="relative w-full aspect-square rounded-2xl overflow-hidden shadow-md bg-[#F7F3E9]/30">
                    {/* Processing Overlay */}
                    {transformMutation.isPending && (
                      <div className="absolute inset-0 bg-white/80 backdrop-blur-sm flex flex-col items-center justify-center z-10">
                        <div className="w-16 h-16 mb-4 relative">
                          <div className="absolute inset-0 rounded-full border-4 border-[#88B2D3]/30 border-t-[#88B2D3] animate-spin"></div>
                        </div>
                        <p className="font-quicksand font-medium text-[#3C4F65]">Transforming Your Image...</p>
                        <p className="text-sm text-[#3C4F65]/70 mt-2">This may take a minute</p>
                      </div>
                    )}
                  
                    {/* Comparison Slider */}
                    <ComparisonSlider 
                      beforeImage={originalImageUrl} 
                      afterImage={transformedImageUrl} 
                    />
                  </div>
                  
                  {/* Image Info & Actions */}
                  <div className="w-full flex items-center justify-between mt-6">
                    <div>
                      <p className="font-medium text-[#3C4F65]">Transform Preview</p>
                      <p className="text-sm text-[#3C4F65]/60">Drag slider to compare</p>
                    </div>
                    <div className="flex space-x-2">
                      <Button
                        size="icon"
                        variant="outline"
                        className="p-2 rounded-full bg-white border border-gray-200 hover:bg-gray-50"
                        title="Download image"
                        disabled={!transformedImageUrl}
                        onClick={handleDownload}
                      >
                        <DownloadIcon className="h-4 w-4 text-[#3C4F65]" />
                      </Button>
                      <Button
                        size="icon"
                        variant="outline"
                        className="p-2 rounded-full bg-white border border-gray-200 hover:bg-gray-50"
                        title="Share image"
                        disabled={!transformedImageUrl}
                      >
                        <Share2Icon className="h-4 w-4 text-[#3C4F65]" />
                      </Button>
                    </div>
                  </div>
                </TabsContent>
                
                <TabsContent value="download" className="p-6">
                  {transformedImageUrl && (
                    <div className="flex flex-col items-center">
                      <div className="mb-6 w-full max-w-md">
                        <img 
                          src={transformedImageUrl} 
                          alt="Transformed image" 
                          className="w-full h-auto rounded-lg shadow-md"
                        />
                      </div>
                      <Button
                        className="bg-[#88B2D3] hover:bg-[#6892B3] text-white rounded-full px-6 py-3"
                        onClick={handleDownload}
                      >
                        <DownloadIcon className="mr-2 h-4 w-4" />
                        Download Image
                      </Button>
                    </div>
                  )}
                </TabsContent>
              </Tabs>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
