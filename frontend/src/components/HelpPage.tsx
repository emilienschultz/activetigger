import { FC } from 'react';

import { PageLayout } from './layout/PageLayout';

export const HelpPage: FC = () => {
  return (
    <PageLayout currentPage="help">
      <div className="container-fluid">
        <div className="row">
          <h2 className="subsection">Documentation</h2>
        </div>
      </div>{' '}
    </PageLayout>
  );
};
