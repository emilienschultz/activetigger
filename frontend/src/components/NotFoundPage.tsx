import { FC } from 'react';
import notfound from '../assets/notfound.png';

import { PageLayout } from './layout/PageLayout';

export const NotFound: FC = () => {
  return (
    <PageLayout currentPage="notfound">
      <div className="container-fluid">
        <div className="row text-center">
          <div className="col-8">
            <h2 className="subsection text-muted mb-4">Not found</h2>
            <img
              src={notfound}
              alt="ActiveTigger"
              className="m-5"
              style={{ width: '500px', height: '500px' }}
            />
          </div>
        </div>
      </div>{' '}
    </PageLayout>
  );
};
