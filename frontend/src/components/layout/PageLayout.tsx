import { FC, ReactNode } from 'react';

import NavBar from './NavBar';
import Notifications from './Notifications';

type PageLayoutProps = { currentPage?: string; children?: ReactNode; projectName?: string | null };
//type PageLayoutProps = PropsWithChildren<{ currentPage?: string;}>

/**
 * PageLayout
 * page layout is a generic component which design an application page HTML layout
 * it should be used by all application page but the static information ones such as home page about..
 * The idea is to avoid repeating navigation/UI custom elements on each page
 */
export const PageLayout: FC<PageLayoutProps> = ({ children, currentPage, projectName }) => {
  return (
    <div>
      <NavBar currentPage={currentPage} projectName={projectName || null} />
      <main>{children}</main>
      <Notifications />
      <footer className="footer mt-auto py-1 bg-primary text-white fixed-bottom">
        <div className="container text-center">
          <i className="fas fa-info-circle"></i>
          <span className="ml-2">CREST / GENES CSS @ IPP © 2024</span>
        </div>
      </footer>
    </div>
  );
};
