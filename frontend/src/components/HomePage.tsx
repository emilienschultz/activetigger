import { FC } from 'react';
import { Link } from 'react-router-dom';

import { LoginForm } from './forms/LoginForm';

export const HomePage: FC = () => {
  return (
    <main className="container-fluid">
      <div className="row">
        <div className="col-0 col-lg-3" />
        <div className="col-12 col-lg-6">
          <h1>Active tigger</h1>
          <h3>Explore & Annotate textual data</h3>
          <LoginForm />
          <div className="general-info">
            <div>Frontend v0.1</div>
            <div>For any information, please contact emilien.schultz [at] ensae.fr</div>
          </div>
        </div>
      </div>
    </main>
  );
};
