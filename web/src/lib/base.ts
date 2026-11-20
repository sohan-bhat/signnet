// Where the site is served from, without a trailing slash: "" at the root of
// a domain, "/signnet" at sohanbhat.me/signnet/. The files in public/ (the
// weights and the sample signs) are loaded from here.
export const BASE = import.meta.env.BASE_URL.replace(/\/$/, "");
