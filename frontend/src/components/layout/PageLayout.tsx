import { FC, ReactNode } from 'react';

import NavBar from './NavBar';

type PageLayoutProps = { currentPage?: string; children?: ReactNode };
//type PageLayoutProps = PropsWithChildren<{ currentPage?: string;}>

export const PageLayout: FC<PageLayoutProps> = ({ children, currentPage }) => {
  return (
    <div>
      <NavBar currentPage={currentPage} />
      <main>{children}</main>
    </div>
  );
};
