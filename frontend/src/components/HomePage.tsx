import { FC } from 'react';
import { Link } from 'react-router-dom';

export const HomePage: FC = () => {
  return (
    <main className="container-fluid">
      <div className="row">
        <div className="col-0 col-lg-3" />
        <div className="col-12 col-lg-6">
          <h1>Bienvenue !</h1>
          <Link to="/login">login</Link>
        </div>
        <div className="col-0 col-lg-3" />
      </div>
    </main>
  );
};
