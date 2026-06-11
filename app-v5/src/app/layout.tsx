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

const TITLE = "Vasu Agrawal — AI Developer";
const DESCRIPTION =
  "Building production LLM systems and agentic AI — from campus chatbots to WhatsApp-native B2B commerce.";

export const metadata: Metadata = {
  title: TITLE,
  description: DESCRIPTION,
  metadataBase: new URL("https://vasuai.dev"),
  openGraph: {
    title: TITLE,
    description: DESCRIPTION,
    url: "/",
    siteName: "VASU_OS 5",
    type: "website",
    images: [{ url: "/og.png", width: 1200, height: 630, alt: TITLE }],
  },
  twitter: {
    card: "summary_large_image",
    title: TITLE,
    description: DESCRIPTION,
    images: ["/og.png"],
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`${plexMono.variable} h-full antialiased`}>
      <body className="min-h-full">
        {/* Pre-paint boot decision (Phase 10): runs synchronously during
            parsing, BEFORE the overlay or hero paint — return visitors and
            reduced-motion users never see a boot flash, and without JS the
            attribute is never set so the overlay stays hidden entirely.
            Must stay the first child of <body>. */}
        <script
          dangerouslySetInnerHTML={{
            __html:
              'try{if(!localStorage.getItem("v5:booted")&&!matchMedia("(prefers-reduced-motion: reduce)").matches)document.documentElement.setAttribute("data-boot","play")}catch(e){}',
          }}
        />
        {children}
      </body>
    </html>
  );
}
