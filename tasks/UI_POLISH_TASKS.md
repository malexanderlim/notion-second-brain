# UI Polish and Logo Integration Tasks

This file outlines UI improvements for the Notion Second Brain application, including logo integration.

## Completed Tasks

- [x] **Responsive Query Input**: Ensured the "Ask your Notion knowledge base..." input field appears on its own line on mobile devices.
    - The `div` wrapper for the input and buttons in `frontend/src/components/layout/MainAppLayout.tsx` was updated to `flex flex-col sm:flex-row gap-2 items-center w-full`.
- [x] **Revert Header Redesign**: The "My Second Brain" title is now back to being left-aligned, with user info top-right.
    - The `<header>` in `frontend/src/components/layout/MainAppLayout.tsx` was restored to `flex justify-between items-center`.
    - The `<h1>` title is now `text-xl font-semibold`.
    - The user info `div` is `flex items-center space-x-2 bg-muted p-2 rounded-lg`, and `LogoutButton` has `className="ml-auto"`.
- [x] **Update Favicon**: Changed the application's favicon to `Notion_Second_Brain.png`.
    - Updated `frontend/index.html`: `<link rel="icon" type="image/png" href="/Notion_Second_Brain.png" />`.
    - User needs to ensure `Notion_Second_Brain.png` is in `frontend/public/`.
- [x] **Add Logo to Main UI**: Added `Notion_Second_Brain.png` above the "Last Synced Entry" text and made its size responsive.
    - Added `<img src="/Notion_Second_Brain.png" ... />` in `frontend/src/components/layout/MainAppLayout.tsx`.
    - Logo is centered, `h-12 w-12` by default (mobile) and `md:h-20 md:w-20` for medium screens and up. Uses `object-contain`.
    - Conditionally rendered: only visible when `!isLoading && !response`.

## In Progress Tasks

- [ ] (No tasks currently in progress for this set of changes)

## Future Tasks

- [ ] Consider adding a dedicated logo component if usage expands.
- [ ] Evaluate logo visibility and UX on various screen sizes and states.

## Implementation Plan

The UI has been updated as per the user's request. The header was reverted to a left-aligned title. The new logo (`Notion_Second_Brain.png`) has been integrated as the site favicon and as a visual element on the main query page. This logo is designed to be unobtrusive, appearing only when the user is about to type a query and disappearing once results are loading or displayed.

### Relevant Files

- `frontend/src/components/layout/MainAppLayout.tsx` - Modified for header reversion, logo addition, and responsive input.
- `frontend/index.html` - Modified to update the favicon.
- `frontend/public/Notion_Second_Brain.png` - New logo file (user to place here). 