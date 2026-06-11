import type { Metadata } from "next";
import { IBM_Plex_Mono } from "next/font/google";
import "./globals.css";

const plexMono = IBM_Plex_Mono({
  variable: "--font-plex-mono",
  subsets: ["latin"],
  // Only the weights actually used: 400 body, 500 medium accents, 700 headings.
  // Every listed weight becomes a preloaded woff2 ahead of the LCP paint.
  weight: ["400", "500", "700"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "Vasu Agrawal — AI Developer",
  description:
    "Building production LLM systems and agentic AI — from campus chatbots to WhatsApp-native B2B commerce.",
  metadataBase: new URL("https://vasuai.dev"),
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`${plexMono.variable} h-full antialiased`}>
      <body className="min-h-full">{children}</body>
    </html>
  );
}
