import { FC } from 'react';

import { LoginForm } from './forms/LoginForm';
import NavBar from './layout/NavBar';

export const HomePage: FC = () => {
  return (
    <>
      <NavBar currentPage="home" />
      <main>
        <LoginForm />
      </main>
    </>
  );
};
